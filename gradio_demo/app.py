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

from inference.models import YOLOWorld
from src.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from src.efficientvit.sam_model_zoo import create_sam_model
import supervision as sv

from src.pipelines.lora_pipeline import LoraMultiConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from src.pipelines.lora_pipeline import revise_regionally_controlnet_forward

CHARACTER_MAN_NAMES = list(character_man.keys())
CHARACTER_WOMAN_NAMES = list(character_woman.keys())
MAX_SEED = np.iinfo(np.int32).max

### Description
title = r"""
<h1 align="center">OMG: Occlusion-friendly Personalized Multi-concept Generation In Diffusion Models</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/' target='_blank'><b>OMG: Occlusion-friendly Personalized Multi-concept Generation In Diffusion Models</b></a>.<br>

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
title={OMG: Occlusion-friendly Personalized Multi-concept Generation In Diffusion Models},
author={},
journal={},
year={}
}
```
"""

tips = r"""
### Usage tips of OMG
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
        print(TEXT_PROMPT + " detected")
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

def build_segment_model(sam_path, device):
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url=sam_path).to(device).eval()
    )
    return yolo_world, sam



def remove_tips():
    return gr.update(visible=False)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def main(device):
    pipe, controller, pipe_concept = build_model_sd(args.pretrained_sdxl_model, args.controlnet_checkpoint, device, prompts_tmp)
    yolo_world, sam = build_segment_model(args.sam_checkpoint, device)



    def generate_image(prompt1, prompt2, prompt3, prompt4, negative_prompt, man, woman, resolution, local_prompt1, local_prompt2, seed):
        try:
            path1 = lorapath_man[man]
            path2 = lorapath_woman[woman]
            pipe_concept.unload_lora_weights()
            pipe_list = build_model_lora(pipe_concept, path1 + "|" + path2)

            input_list = [prompt1, prompt2, prompt3, prompt4]
            output_list = []

            width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])

            kwargs = {
                'height': height,
                'width': width,
            }

            for prompt in input_list:
                if prompt!='':
                    input_prompt = []
                    p = 'Close-up photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed.'
                    input_prompt.append([p.replace("{prompt}", prompt), p.replace("{prompt}", prompt)])
                    input_prompt.append([(local_prompt1, character_man.get(man)[1]), (local_prompt2, character_woman.get(woman)[1])])

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

                    mask1 = predict_mask(yolo_world, sam, image[0], 'man', confidence=0.2, threshold=0.5)
                    mask2 = predict_mask(yolo_world, sam, image[0], 'woman', confidence=0.2, threshold=0.5)

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
            usage_tips = gr.Markdown(label="Usage tips of OMG", value=tips, visible=False)

        # character choose
        with gr.Row():
            man = gr.Dropdown(label="Character 1 selection", choices=CHARACTER_MAN_NAMES, value="Harry Potter (identifier: Harry Potter)")
            woman = gr.Dropdown(label="Character 2 selection", choices=CHARACTER_WOMAN_NAMES, value="Hermione Granger (identifier: Hermione Granger)")
            res = gr.Dropdown(label="Image Resolution", choices=["1024*1024", "1440*728"], value="1024*1024")

        with gr.Row():
            local_prompt1 = gr.Textbox(label="Character1_prompt",
                                info="Describe the Character 1, this prompt should include the identifier of character 1",
                                value="Close-up photo of the Harry Potter, 35mm photograph, film, professional, 4k, highly detailed.")
            local_prompt2 = gr.Textbox(label="Character2_prompt",
                                       info="Describe the Character 2, this prompt should include the identifier of character2",
                                       value="Close-up photo of the Hermione Granger, 35mm photograph, film, professional, 4k, highly detailed.")

        # prompt
        with gr.Column():
            prompt = gr.Textbox(label="Prompt 1",
                                info="Give a simple prompt to describe the first image content",
                                placeholder="Required",
                                value="the man and woman's surprised expressions as they accidentally discover a mysterious island while on vacation by the sea")
            prompt2 = gr.Textbox(label="Prompt 2",
                                 info="Give a simple prompt to describe the second image content",
                                 placeholder="optional",
                                 value="")
            prompt3 = gr.Textbox(label="Prompt 3",
                                 info="Give a simple prompt to describe the third image content",
                                 placeholder="optional",
                                 value="")
            prompt4 = gr.Textbox(label="Prompt 4",
                                 info="Give a simple prompt to describe the fourth image content",
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
            inputs=[prompt, prompt2, prompt3, prompt4, negative_prompt, man, woman, res, local_prompt1, local_prompt2, seed],
            outputs=[gallery, gallery2, gallery3, gallery4]
        )
    demo.launch(server_name='0.0.0.0',server_port=7861, debug=True)

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_sdxl_model', default='./checkpoint/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--controlnet_checkpoint', default='./checkpoint/controlnet-openpose-sdxl-1.0', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/xl1.pt', type=str)
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