import sys
sys.path.append('./')
import gradio as gr
import random
import numpy as np
from gradio_demo.character_template import character_man, lorapath_man
from gradio_demo.character_template import character_woman, lorapath_woman
import torch
import os
from typing import Tuple, List
import copy
import argparse

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, predict
from segment_anything import build_sam, SamPredictor
import groundingdino.datasets.transforms as T

from src.pipelines.lora_pipeline import LoraMultiConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from src.pipelines.lora_pipeline import revise_regionally_controlnet_forward

CHARACTER_MAN_NAMES = list(character_man.keys())
CHARACTER_WOMAN_NAMES = list(character_woman.keys())
MAX_SEED = np.iinfo(np.int32).max

### Description
title = r"""
<h1 align="center">MMC: Mastering Multiple Concepts in One Frame</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/' target='_blank'><b>MMC: Mastering Multiple Concepts in One Frame</b></a>.<br>

How to use:<br>
1. Select two characters.
2. Enter a text prompt as done in normal text-to-image models.
3. Click the <b>Submit</b> button to start customizing.
4. Enjoy the generated image😊!
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{,
title={MMC: Mastering Multiple Concepts in One Frame},
author={},
journal={},
year={}
}
```
"""

tips = r"""
### Usage tips of MMC
1. Input text prompts to describe a man and a woman
"""

css = '''
.gradio-container {width: 85% !important}
'''

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
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def predict_mask(groundingdino_model, sam_predictor, image, TEXT_PROMPT, BOX_TRESHOLD = 0.3, TEXT_TRESHOLD = 0.25):
    image_source, image = load_image(image)
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
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

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    args = SLConfig.fromfile(ckpt_config_filename)
    model = build_model(args)
    args.device = device

    checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(filename, log))
    _ = model.eval()
    return model

def build_model_sd(pretrained_model, controlnet_path, device, prompts):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
    pipe = LoraMultiConceptPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to(device)
    controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.}, self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device, dtype=torch.float16)
    revise_regionally_controlnet_forward(pipe.unet, controller)
    pipe_concept = StableDiffusionXLPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16,
                                                             variant="fp16").to(device)
    return pipe, controller, pipe_concept

def build_model_lora(pipe_concept, lora_paths):
    pipe_list = []
    for lora_path in lora_paths.split('|'):
        adapter_name = lora_path.split('/')[-1].split('.')[0]
        pipe_concept.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=adapter_name)
        pipe_concept.enable_xformers_memory_efficient_attention()
        pipe_list.append(adapter_name)
    return pipe_list

def build_segment_model(ckpt_repo_id, sam_checkpoint):
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return groundingdino_model, sam_predictor



def remove_tips():
    return gr.update(visible=False)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def main(device):
    pipe, controller, pipe_concept = build_model_sd(args.pretrained_sdxl_model, args.controlnet_checkpoint, device, prompts_tmp)
    groundingdino_model, sam_predictor = build_segment_model(args.dino_checkpoint, args.sam_checkpoint)

    width, height = 1024, 1024
    kwargs = {
        'height': height,
        'width': width,
    }

    def generate_image(prompt1, prompt2, prompt3, prompt4, negative_prompt, man, woman, seed):
        try:
            path1 = lorapath_man[man]
            path2 = lorapath_woman[woman]
            pipe_concept.unload_lora_weights()
            pipe_list = build_model_lora(pipe_concept, path1 + "|" + path2)

            input_list = [prompt1, prompt2, prompt3, prompt4]
            output_list = []

            for prompt in input_list:
                if prompt!='':
                    input_prompt = []
                    p = 'Close-up photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed.'
                    input_prompt.append([p.replace("{prompt}", prompt), p.replace("{prompt}", prompt)])
                    input_prompt.append([character_man.get(man), character_woman.get(woman)])

                    controller.reset()
                    image = sample_image(
                        pipe,
                        input_prompt=input_prompt,
                        concept_models=pipe_concept,
                        input_neg_prompt=[negative_prompt] * len(input_prompt),
                        generator=torch.Generator(device).manual_seed(seed),
                        controller=controller,
                        stage=1,
                        lora_list=pipe_list,
                        **kwargs)

                    controller.reset()

                    mask1 = predict_mask(groundingdino_model, sam_predictor, image[0], 'man')[0].squeeze(0)
                    mask2 = predict_mask(groundingdino_model, sam_predictor, image[0], 'woman')[0].squeeze(0)

                    image = sample_image(
                        pipe,
                        input_prompt=input_prompt,
                        concept_models=pipe_concept,
                        input_neg_prompt=[negative_prompt] * len(input_prompt),
                        generator=torch.Generator(device).manual_seed(seed),
                        controller=controller,
                        stage=2,
                        region_masks=[mask1, mask2],
                        lora_list=pipe_list,
                        **kwargs)
                    output_list.append(image[1])
                else:
                    output_list.append(None)
            return output_list
        except:
            print("error")
            return None, None, None, None

    with gr.Blocks(css=css) as demo:
        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            gallery = gr.Image(label="Generated Images", height=512, width=512)
            gallery2 = gr.Image(label="Generated Images", height=512, width=512)
            gallery3 = gr.Image(label="Generated Images", height=512, width=512)
            gallery4 = gr.Image(label="Generated Images", height=512, width=512)
            usage_tips = gr.Markdown(label="Usage tips of MMC", value=tips, visible=False)

        # character choose
        with gr.Row():
            man = gr.Dropdown(label="Character 1 selection", choices=CHARACTER_MAN_NAMES, value="Harry Potter")
            woman = gr.Dropdown(label="Character 2 selection", choices=CHARACTER_WOMAN_NAMES, value="Hermione Granger")



        # prompt
        with gr.Column():
            prompt = gr.Textbox(label="Prompt 1",
                                info="Give a simple prompt to describe the image content",
                                placeholder="Required",
                                value="the man and woman's surprised expressions as they accidentally discover a mysterious island while on vacation by the sea")
            prompt2 = gr.Textbox(label="Prompt 2",
                                 info="Give a simple prompt to describe the image content",
                                 placeholder="optional",
                                 value="")
            prompt3 = gr.Textbox(label="Prompt 3",
                                 info="Give a simple prompt to describe the image content",
                                 placeholder="optional",
                                 value="")
            prompt4 = gr.Textbox(label="Prompt 4",
                                 info="Give a simple prompt to describe the image content",
                                 placeholder="optional",
                                 value="")

        with gr.Accordion(open=False, label="Advanced Options"):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )
            negative_prompt = gr.Textbox(label="Negative Prompt",
                                placeholder="noisy, blurry, soft, deformed, ugly",
                                value="noisy, blurry, soft, deformed, ugly")
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        submit = gr.Button("Submit", variant="primary")

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[prompt, prompt2, prompt3, prompt4, negative_prompt, man, woman, seed],
            outputs=[gallery, gallery2, gallery3, gallery4]
        )
    demo.launch(server_name='0.0.0.0',server_port=7861, debug=True)

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_sdxl_model', default='./checkpoint/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--controlnet_checkpoint', default='./checkpoint/controlnet-openpose-sdxl-1.0', type=str)
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--prompt', default='Close-up photo of the cool man and beautiful woman in surprised expressions as they accidentally discover a mysterious island while on vacation by the sea, 35mm photograph, film, professional, 4k, highly detailed.', type=str)
    parser.add_argument('--negative_prompt', default='noisy, blurry, soft, deformed, ugly', type=str)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--suffix', default='', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    prompts = [args.prompt]*2
    prompts_tmp = copy.deepcopy(prompts)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(device)