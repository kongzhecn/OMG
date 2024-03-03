import argparse
import hashlib
import json
import os.path
from typing import Tuple, List
import torch
import copy

from inference.models import YOLOWorld
from src.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from src.efficientvit.sam_model_zoo import create_sam_model
import supervision as sv


from src.pipelines.lora_pipeline import LoraMultiConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from src.pipelines.lora_pipeline import revise_regionally_controlnet_forward

import numpy as np
from torchvision.utils import save_image

def sample_image(pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    concept_models=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    controller=None,
    stage=None,
    region_masks=None,
    lora_list = None,
    **extra_kargs
):
    images = pipe(
        prompt=input_prompt,
        concept_models=concept_models,
        negative_prompt=input_neg_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": 0.8},
        controller=controller,
        stage=stage,
        region_masks=region_masks,
        lora_list=lora_list,
        **extra_kargs).images

    return images

def load_image(image_source) -> Tuple[np.array, torch.Tensor]:
    image = np.asarray(image_source)
    return image


def predict_mask(yolo_world, sam, image, TEXT_PROMPT, confidence = 0.2, threshold = 0.5):
    image_source = load_image(image)
    yolo_world.set_classes([TEXT_PROMPT])
    results = yolo_world.infer(image_source, confidence=confidence)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=threshold
    )
    masks = None
    if len(detections) != 0:
        sam.set_image(image_source, image_format="RGB")
        masks, _, _ = sam.predict(box=detections.xyxy[0], multimask_output=False)
        masks = torch.from_numpy(masks.squeeze())
    return masks

def prepare_text(prompt, region_prompts):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    region_collection = []

    regions = region_prompts.split('|')

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')

        region_collection.append((prompt_region, neg_prompt_region))
    return (prompt, region_collection)


def build_model_sd(pretrained_model, controlnet_path, device, prompts, lora_paths):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
    pipe = LoraMultiConceptPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to(device)
    controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.}, self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device, dtype=torch.float16)
    revise_regionally_controlnet_forward(pipe.unet, controller)

    pipe_concept = StableDiffusionXLPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16, variant="fp16").to(device)
    pipe_concept.enable_xformers_memory_efficient_attention()

    pipe_list = []
    for lora_path in lora_paths.split('|'):
        adapter_name = lora_path.split('/')[-1].split('.')[0]
        pipe_concept.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=adapter_name)
        pipe_list.append(adapter_name)
    return pipe, controller, pipe_concept, pipe_list

def build_segment_model(sam_path, device):
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url=sam_path).to(device).eval()
    )
    return yolo_world, sam


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_sdxl_model', default='./checkpoint/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--controlnet_checkpoint', default='./checkpoint/controlnet-openpose-sdxl-1.0', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/xl1.pt', type=str)
    parser.add_argument('--save_dir', default='results/lora', type=str)
    parser.add_argument('--prompt', default='Close-up photo of the cool man and beautiful woman in surprised expressions as they accidentally discover a mysterious island while on vacation by the sea, 35mm photograph, film, professional, 4k, highly detailed.', type=str)
    parser.add_argument('--negative_prompt', default='noisy, blurry, soft, deformed, ugly', type=str)
    parser.add_argument('--prompt_rewrite',
                        default='[Close-up photo of the Harry Potter in surprised expressions as he wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-*'
                                '-[noisy, blurry, soft, deformed, ugly]|'
                                '[Close-up photo of the Hermione Granger in surprised expressions as she wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-'
                                '*-[noisy, blurry, soft, deformed, ugly]',
                        type=str)
    parser.add_argument('--lora_path', default='./checkpoint/lora/Harry_Potter.safetensors|./checkpoint/lora/Hermione_Granger.safetensors', type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--suffix', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prompts = [args.prompt]*2
    prompts_tmp = copy.deepcopy(prompts)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pipe, controller, pipe_concepts, pipe_list = build_model_sd(args.pretrained_sdxl_model, args.controlnet_checkpoint, device, prompts_tmp, args.lora_path)
    yolo_world, sam = build_segment_model(args.sam_checkpoint, device)

    width, height = 1024, 1024
    kwargs = {
        'height': height,
        'width': width,
    }

    prompts_rewrite = [args.prompt_rewrite]
    input_prompt = [prepare_text(p, p_w) for p, p_w in zip(prompts, prompts_rewrite)]
    input_prompt = [prompts, input_prompt[0][1]]

    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        concept_models=pipe_concepts,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        controller=controller,
        stage=1,
        lora_list=pipe_list,
        **kwargs)

    controller.reset()

    mask1 = predict_mask(yolo_world, sam, image[0], 'man', confidence = 0.2, threshold = 0.5)
    mask2 = predict_mask(yolo_world, sam, image[0], 'woman', confidence = 0.2, threshold = 0.5)


    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        concept_models=pipe_concepts,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        controller=controller,
        stage=2,
        region_masks=[mask1, mask2],
        lora_list=pipe_list,
        **kwargs)



    configs = [
        f'pretrained_model: {args.pretrained_sdxl_model}\n',
        f'context_prompt: {args.prompt}\n', f'neg_context_prompt: {args.negative_prompt}\n',
        f'prompt_rewrite: {args.prompt_rewrite}\n'
    ]
    hash_code = hashlib.sha256(''.join(configs).encode('utf-8')).hexdigest()[:8]

    save_name = f'**---{args.suffix}---{hash_code}.png'
    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    save_path = os.path.join(save_dir, save_name)
    save_config_path = os.path.join(save_dir, save_name.replace('.png', '.txt'))
    attention_path = os.path.join(save_dir, save_name).replace('**', 'att')

    os.makedirs(save_dir, exist_ok=True)
    print(f'save to: {save_dir}')
    stage1_path = os.path.join(save_dir, 'stage-1.png')
    stage2_path = os.path.join(save_dir, 'stage-2.png')
    image[0].save(stage1_path)
    image[1].save(stage2_path)


    with open(save_config_path, 'w') as fw:
        fw.writelines(configs)