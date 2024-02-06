import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.utils.import_utils import is_invisible_watermark_available

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers import StableDiffusionXLControlNetPipeline
from PIL import Image
from torchvision.transforms.functional import to_tensor
from einops import rearrange
from torch import einsum
import math
from torchvision.utils import save_image
from diffusers.utils import load_image
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class RegionControlNet_AttnProcessor:
    def __init__(self, attention_op=None, controller=None, place_in_unet=None):
        self.attention_op = attention_op
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            **cross_attention_kwargs
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross = True
        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_probs, value)


        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def revise_regionally_controlnet_forward(unet, controller):
    def change_forward(unet, count, place_in_unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention':
                layer.set_processor(RegionControlNet_AttnProcessor(controller=controller, place_in_unet=place_in_unet))
                if 'attn2' in name:
                    count += 1
            else:
                count = change_forward(layer, count, place_in_unet)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0, "down")
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx, "up")
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx, "mid")
    print(f'Number of attention layer registered {cross_attention_idx}')
    controller.num_att_layers = cross_attention_idx*2

class InstantidMultiConceptPipeline(StableDiffusionXLControlNetPipeline):
    # leave controlnet out on purpose because it iterates with unet
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "feature_extractor",
        "image_encoder",
    ]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
    ):
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        controller=None,
        concept_models=None,
        indices_to_alter=None,
        face_app=None,
        stage=None,
        region_masks=None,
        **kwargs,
    ):
        # revise_regionally_controlnet_forward(self.unet, controller)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        batch_size = 2

        device = self._execution_device

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3.1 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        global_prompt = prompt[0]
        global_negative_prompt = negative_prompt
        region_prompts = [pt[0] for pt in prompt[1]]
        region_negative_prompts = [pt[1] for pt in prompt[1]]
        ref_images = [pt[2] for pt in prompt[1]]

        concat_prompts = global_prompt + region_prompts
        concat_negative_prompts = global_negative_prompt + region_negative_prompts

        (
            concat_prompt_embeds,
            concat_negative_prompt_embeds,
            concat_pooled_prompt_embeds,
            concat_negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            concat_prompts,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            concat_negative_prompts,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        prompt_embeds = concat_prompt_embeds[:2]
        negative_prompt_embeds = concat_negative_prompt_embeds[:2]
        pooled_prompt_embeds = concat_pooled_prompt_embeds[:2]
        negative_pooled_prompt_embeds = concat_negative_pooled_prompt_embeds[:2]

        region_prompt_embeds_list = []
        region_add_text_embeds_list = []
        for region_prompt_embeds, region_negative_prompt_embeds, region_pooled_prompt_embeds, region_negative_pooled_prompt_embeds in zip(concat_prompt_embeds[2:], concat_negative_prompt_embeds[2:], concat_pooled_prompt_embeds[2:], concat_negative_pooled_prompt_embeds[2:]):
            region_prompt_embeds_list.append(
                torch.concat([region_negative_prompt_embeds.unsqueeze(0), region_prompt_embeds.unsqueeze(0)], dim=0).to(concept_models._execution_device))
            region_add_text_embeds_list.append(
                torch.concat([region_negative_pooled_prompt_embeds.unsqueeze(0), region_pooled_prompt_embeds.unsqueeze(0)], dim=0).to(concept_models._execution_device))


        if stage==2:
            mask_list = [mask.float().to(dtype=prompt_embeds.dtype, device=device) for mask in region_masks]
            image_embedding_list = get_face_embedding(face_app, ref_images)
            image_prompt_image_emb_list = []
            for image_embeds in image_embedding_list:
                prompt_image_emb = concept_models._encode_prompt_image_emb(image_embeds,
                                                             concept_models._execution_device,
                                                             num_images_per_prompt,
                                                             concept_models.unet.dtype,
                                                             True)
                image_prompt_image_emb_list.append(prompt_image_emb)



        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel) and image is not None:
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=1 * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel) and image is not None:
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size//2 * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.1 repeat latent
        latents = torch.cat([latents, latents.clone()])

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        add_time_ids_list = []
        region_add_time_ids = concept_models._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim)
        for _ in range(len(prompt[1])):
            add_time_ids_list.append(torch.concat([region_add_time_ids, region_add_time_ids], dim=0).to(concept_models._execution_device))

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_controlnet_compiled = is_compiled_module(self.controlnet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
        # hyper-parameters
        scale_range = np.linspace(1, 0.5, len(self.scheduler.timesteps))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]


                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if i > 15 and stage == 2:
                    region_mask = self.get_region_mask(mask_list, noise_pred.shape[2], noise_pred.shape[3])
                    edit_noise = torch.concat([noise_pred[1:2], noise_pred[3:4]], dim=0)
                    new_noise_pred = torch.zeros_like(edit_noise)
                    new_noise_pred[:, :, region_mask == 0] = edit_noise[:, :, region_mask == 0]
                    replace_ratio = 1.0
                    new_noise_pred[:, :, region_mask != 0] = (1 - replace_ratio) * edit_noise[:, :, region_mask != 0]

                    for region_prompt_embeds, region_add_text_embeds, region_add_time_ids, concept_mask, region_prompt, region_prompt_image_emb in zip(region_prompt_embeds_list, region_add_text_embeds_list, add_time_ids_list, mask_list, region_prompts, image_prompt_image_emb_list):

                        concept_mask = F.interpolate(concept_mask.unsqueeze(0).unsqueeze(0),
                                                     size=(noise_pred.shape[2], noise_pred.shape[3]),
                                                     mode='nearest').squeeze().to(dtype=noise_pred.dtype, device=concept_models._execution_device)

                        region_latent_model_input = latent_model_input[3:4].clone().to(concept_models._execution_device)

                        region_latent_model_input = torch.cat([region_latent_model_input] * 2)
                        region_added_cond_kwargs = {"text_embeds": region_add_text_embeds,
                                                    "time_ids": region_add_time_ids}

                        if image is not None:
                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                region_latent_model_input,
                                t,
                                encoder_hidden_states=region_prompt_image_emb,
                                controlnet_cond=image,
                                conditioning_scale=cond_scale,
                                guess_mode=guess_mode,
                                added_cond_kwargs=region_added_cond_kwargs,
                                return_dict=False,
                            )

                            if guess_mode and self.do_classifier_free_guidance:
                                # Infered ControlNet only for the conditional batch.
                                # To apply the output of ControlNet to both the unconditional and conditional batches,
                                # add 0 to the unconditional batch to keep it unchanged.
                                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in
                                                          down_block_res_samples]
                                mid_block_res_sample = torch.cat(
                                    [torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                        else:
                            down_block_res_samples = None
                            mid_block_res_sample = None

                        region_encoder_hidden_states = torch.cat([region_prompt_embeds, region_prompt_image_emb], dim=1)

                        region_noise_pred = concept_models.unet(
                            region_latent_model_input,
                            t,
                            encoder_hidden_states=region_encoder_hidden_states,
                            cross_attention_kwargs=None,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            added_cond_kwargs=region_added_cond_kwargs,
                            return_dict=False,
                        )[0]


                        new_noise_pred = new_noise_pred.to(concept_models._execution_device)
                        new_noise_pred[:, :, concept_mask==1] += replace_ratio * (region_noise_pred[:, :, concept_mask==1] / (concept_mask.reshape(1, 1, *concept_mask.shape)[:, :, concept_mask==1].to(region_noise_pred.device)))
                        # print(region_noise_pred.shape)


                    new_noise_pred = new_noise_pred.to(noise_pred.device)
                    noise_pred[1, :, :, :] = new_noise_pred[0]
                    noise_pred[3, :, :, :] = new_noise_pred[1]
                    # controller.between_steps()

                    # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # manually for max memory savings
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def check_image(self, image, prompt, prompt_embeds):
        pass

    def get_region_mask(self, mask_list, feat_height, feat_width):
        exclusive_mask = torch.zeros((feat_height, feat_width))
        for mask in mask_list:
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(feat_height, feat_width),
                                 mode='nearest').squeeze().to(dtype=exclusive_mask.dtype, device=exclusive_mask.device)
            exclusive_mask = ((mask == 1) | (exclusive_mask == 1)).to(dtype=mask.dtype)
        return exclusive_mask

def get_face_embedding(face_app, ref_images):
    emb_list = []
    for img_path in ref_images:
        face_image = load_image(img_path)

        # prepare face emb
        face_info = face_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[0]  # only use the maximum face
        face_emb = face_info['embedding']
        emb_list.append(face_emb)
        # face_kps = draw_kps(face_image, face_info['kps'])
    return emb_list