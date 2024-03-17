import sys
sys.path.append('./')
import gradio as gr
import random
import numpy as np
from gradio_demo.character_template import character_man, lorapath_man
from gradio_demo.character_template import character_woman, lorapath_woman
from gradio_demo.character_template import styles, lorapath_styles
import torch
import os
from typing import Tuple, List
import copy
import argparse
from diffusers.utils import load_image
import cv2
from PIL import Image, ImageOps
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.body import Body

try:
    from inference.models import YOLOWorld
    from src.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from src.efficientvit.sam_model_zoo import create_sam_model
    import supervision as sv
except:
    print("YoloWorld can not be load")

try:
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, predict
    from segment_anything import build_sam, SamPredictor
    import groundingdino.datasets.transforms as T
except:
    print("groundingdino can not be load")

from src.pipelines.lora_pipeline import LoraMultiConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from src.pipelines.lora_pipeline import revise_regionally_controlnet_forward

CHARACTER_MAN_NAMES = list(character_man.keys())
CHARACTER_WOMAN_NAMES = list(character_woman.keys())
STYLE_NAMES = list(styles.keys())
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
    styleL=None,
    **extra_kargs
):

    spatial_condition = extra_kargs.pop('spatial_condition')
    if spatial_condition is not None:
        spatial_condition_input = [spatial_condition] * len(input_prompt)
    else:
        spatial_condition_input = None

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
        styleL=styleL,
        image=spatial_condition_input,
        **extra_kargs).images

    return images

def load_image_yoloworld(image_source) -> Tuple[np.array, torch.Tensor]:
    image = np.asarray(image_source)
    return image

def load_image_dino(image_source) -> Tuple[np.array, torch.Tensor]:
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

def predict_mask(segmentmodel, sam, image, TEXT_PROMPT, segmentType, confidence = 0.2, threshold = 0.5):
    if segmentType=='GroundingDINO':
        image_source, image = load_image_dino(image)
        boxes, logits, phrases = predict(
            model=segmentmodel,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=0.3,
            text_threshold=0.25
        )
        sam.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
        masks, _, _ = sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks=masks[0].squeeze(0)
    else:
        image_source = load_image_yoloworld(image)
        segmentmodel.set_classes([TEXT_PROMPT])
        results = segmentmodel.infer(image_source, confidence=confidence)
        detections = sv.Detections.from_inference(results).with_nms(
            class_agnostic=True, threshold=threshold
        )
        masks = None
        if len(detections) != 0:
            print(TEXT_PROMPT + " detected!")
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
    controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.}, self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device, dtype=torch.float16, width=1024//32, height=1024//32)
    revise_regionally_controlnet_forward(pipe.unet, controller)
    pipe_concept = StableDiffusionXLPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16,
                                                             variant="fp16").to(device)
    return pipe, controller, pipe_concept

def build_model_lora(pipe_concept, lora_paths, style_path, condition, args, pipe):
    pipe_list = []
    if condition == "Human pose":
        controlnet = ControlNetModel.from_pretrained(args.openpose_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.controlnet = controlnet
    elif condition == "Canny Edge":
        controlnet = ControlNetModel.from_pretrained(args.canny_checkpoint, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.controlnet = controlnet
    elif condition == "Depth":
        controlnet = ControlNetModel.from_pretrained(args.depth_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.controlnet = controlnet

    if style_path is not None and os.path.exists(style_path):
        pipe_concept.load_lora_weights(style_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')
        pipe.load_lora_weights(style_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')

    for lora_path in lora_paths.split('|'):
        adapter_name = lora_path.split('/')[-1].split('.')[0]
        pipe_concept.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name=adapter_name)
        pipe_concept.enable_xformers_memory_efficient_attention()
        pipe_list.append(adapter_name)
    return pipe_list

def build_yolo_segment_model(sam_path, device):
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url=sam_path).to(device).eval()
    )
    return yolo_world, sam

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    args = SLConfig.fromfile(ckpt_config_filename)
    model = build_model(args)
    args.device = device

    checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(filename, log))
    _ = model.eval()
    return model

def build_dino_segment_model(ckpt_repo_id, sam_checkpoint):
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return groundingdino_model, sam_predictor

def resize_and_center_crop(image, output_size=(1024, 576)):
    width, height = image.size
    aspect_ratio = width / height
    new_height = output_size[1]
    new_width = int(aspect_ratio * new_height)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    if new_width < output_size[0] or new_height < output_size[1]:
        padding_color = "gray"
        resized_image = ImageOps.expand(resized_image,
                                        ((output_size[0] - new_width) // 2,
                                         (output_size[1] - new_height) // 2,
                                         (output_size[0] - new_width + 1) // 2,
                                         (output_size[1] - new_height + 1) // 2),
                                        fill=padding_color)

    left = (resized_image.width - output_size[0]) / 2
    top = (resized_image.height - output_size[1]) / 2
    right = (resized_image.width + output_size[0]) / 2
    bottom = (resized_image.height + output_size[1]) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image

def main(device, segment_type):
    pipe, controller, pipe_concept = build_model_sd(args.pretrained_sdxl_model, args.openpose_checkpoint, device, prompts_tmp)

    if segment_type == 'GroundingDINO':
        detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)
    else:
        detect_model, sam = build_yolo_segment_model(args.efficientViT_checkpoint, device)

    resolution_list = ["1440*728",
                       "1344*768",
                       "1216*832",
                       "1152*896",
                       "1024*1024",
                       "896*1152",
                       "832*1216",
                       "768*1344",
                       "728*1440"]
    ratio_list = [1440 / 728, 1344 / 768, 1216 / 832, 1152 / 896, 1024 / 1024, 896 / 1152, 832 / 1216, 768 / 1344,
                  728 / 1440]
    condition_list = ["None",
                      "Human pose",
                      "Canny Edge",
                      "Depth"]

    depth_estimator = DPTForDepthEstimation.from_pretrained(args.dpt_checkpoint).to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained(args.dpt_checkpoint)
    body_model = Body(args.pose_detector_checkpoint)
    openpose = OpenposeDetector(body_model)

    def remove_tips():
        return gr.update(visible=False)

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def get_humanpose(img):
        openpose_image = openpose(img)
        return openpose_image

    def get_cannyedge(image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image

    def get_depth(image):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def generate_image(prompt1, negative_prompt, man, woman, resolution, local_prompt1, local_prompt2, seed, condition, condition_img1, style):
        try:
            path1 = lorapath_man[man]
            path2 = lorapath_woman[woman]
            pipe_concept.unload_lora_weights()
            pipe.unload_lora_weights()
            pipe_list = build_model_lora(pipe_concept, path1 + "|" + path2, lorapath_styles[style], condition, args, pipe)

            if lorapath_styles[style] is not None and os.path.exists(lorapath_styles[style]):
                styleL = True
            else:
                styleL = False

            input_list = [prompt1]
            condition_list = [condition_img1]
            output_list = []

            width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])

            kwargs = {
                'height': height,
                'width': width,
            }

            for prompt, condition_img in zip(input_list, condition_list):
                if prompt!='':
                    input_prompt = []
                    p = '{prompt}, 35mm photograph, film, professional, 4k, highly detailed.'
                    if styleL:
                        p = styles[style] + p
                    input_prompt.append([p.replace("{prompt}", prompt), p.replace("{prompt}", prompt)])
                    if styleL:
                        input_prompt.append([(styles[style] + local_prompt1, character_man.get(man)[1]),
                                             (styles[style] + local_prompt2, character_woman.get(woman)[1])])
                    else:
                        input_prompt.append([(local_prompt1, character_man.get(man)[1]),
                                             (local_prompt2, character_woman.get(woman)[1])])

                    if condition == 'Human pose' and condition_img is not None:
                        index = ratio_list.index(
                            min(ratio_list, key=lambda x: abs(x - condition_img.shape[1] / condition_img.shape[0])))
                        resolution = resolution_list[index]
                        width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])
                        kwargs['height'] = height
                        kwargs['width'] = width
                        condition_img = resize_and_center_crop(Image.fromarray(condition_img), (width, height))
                        spatial_condition = get_humanpose(condition_img)
                    elif condition == 'Canny Edge' and condition_img is not None:
                        index = ratio_list.index(
                            min(ratio_list, key=lambda x: abs(x - condition_img.shape[1] / condition_img.shape[0])))
                        resolution = resolution_list[index]
                        width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])
                        kwargs['height'] = height
                        kwargs['width'] = width
                        condition_img = resize_and_center_crop(Image.fromarray(condition_img), (width, height))
                        spatial_condition = get_cannyedge(condition_img)
                    elif condition == 'Depth' and condition_img is not None:
                        index = ratio_list.index(
                            min(ratio_list, key=lambda x: abs(x - condition_img.shape[1] / condition_img.shape[0])))
                        resolution = resolution_list[index]
                        width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])
                        kwargs['height'] = height
                        kwargs['width'] = width
                        condition_img = resize_and_center_crop(Image.fromarray(condition_img), (width, height))
                        spatial_condition = get_depth(condition_img)
                    else:
                        spatial_condition = None

                    kwargs['spatial_condition'] = spatial_condition
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
                        styleL=styleL,
                        **kwargs)

                    controller.reset()
                    if pipe.tokenizer("man")["input_ids"][1] in pipe.tokenizer(args.prompt)["input_ids"][1:-1]:
                        mask1 = predict_mask(detect_model, sam, image[0], 'man', args.segment_type, confidence=0.15,
                                             threshold=0.5)
                    else:
                        mask1 = None

                    if pipe.tokenizer("woman")["input_ids"][1] in pipe.tokenizer(args.prompt)["input_ids"][1:-1]:
                        mask2 = predict_mask(detect_model, sam, image[0], 'woman', args.segment_type, confidence=0.15,
                                             threshold=0.5)
                    else:
                        mask2 = None

                    if mask1 is None and mask2 is None:
                        output_list.append(image[1])
                    else:
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
                            styleL=styleL,
                            **kwargs)
                        output_list.append(image[1])
                else:
                    output_list.append(None)
            output_list.append(spatial_condition)
            return output_list
        except:
            print("error")
            return

    def get_local_value_man(input):
        return character_man[input][0]

    def get_local_value_woman(input):
        return character_woman[input][0]


    with gr.Blocks(css=css) as demo:
        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            gallery = gr.Image(label="Generated Images", height=512, width=512)
            gen_condition = gr.Image(label="Spatial Condition", height=512, width=512)
            usage_tips = gr.Markdown(label="Usage tips of OMG", value=tips, visible=False)

        with gr.Row():
            condition_img1 = gr.Image(label="Input an RGB image for condition", height=128, width=128)

        # character choose
        with gr.Row():
            man = gr.Dropdown(label="Character 1 selection", choices=CHARACTER_MAN_NAMES, value="Chris Evans (identifier: Chris Evans)")
            woman = gr.Dropdown(label="Character 2 selection", choices=CHARACTER_WOMAN_NAMES, value="Taylor Swift (identifier: TaylorSwift)")
            resolution = gr.Dropdown(label="Image Resolution (width*height)", choices=resolution_list, value="1024*1024")
            condition = gr.Dropdown(label="Input condition type", choices=condition_list, value="None")
            style = gr.Dropdown(label="style", choices=STYLE_NAMES, value="None")

        with gr.Row():
            local_prompt1 = gr.Textbox(label="Character1_prompt",
                                info="Describe the Character 1, this prompt should include the identifier of character 1",
                                value="Close-up photo of the Chris Evans, 35mm photograph, film, professional, 4k, highly detailed.")
            local_prompt2 = gr.Textbox(label="Character2_prompt",
                                       info="Describe the Character 2, this prompt should include the identifier of character2",
                                       value="Close-up photo of the TaylorSwift, 35mm photograph, film, professional, 4k, highly detailed.")

        man.change(get_local_value_man, man, local_prompt1)
        woman.change(get_local_value_woman, woman, local_prompt2)

        # prompt
        with gr.Column():
            prompt = gr.Textbox(label="Prompt 1",
                                info="Give a simple prompt to describe the first image content",
                                placeholder="Required",
                                value="close-up shot, photography, a man and a woman on the street, facing the camera smiling")


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
            inputs=[prompt, negative_prompt, man, woman, resolution, local_prompt1, local_prompt2, seed, condition, condition_img1, style],
            outputs=[gallery, gen_condition]
        )
    demo.launch(server_name='0.0.0.0',server_port=7861, debug=True)

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_sdxl_model', default='./checkpoint/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--openpose_checkpoint', default='./checkpoint/controlnet-openpose-sdxl-1.0', type=str)
    parser.add_argument('--canny_checkpoint', default='./checkpoint/controlnet-canny-sdxl-1.0', type=str)
    parser.add_argument('--depth_checkpoint', default='./checkpoint/controlnet-depth-sdxl-1.0', type=str)
    parser.add_argument('--efficientViT_checkpoint', default='./checkpoint/sam/xl1.pt', type=str)
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--dpt_checkpoint', default='./checkpoint/dpt-hybrid-midas', type=str)
    parser.add_argument('--pose_detector_checkpoint', default='./checkpoint/ControlNet/annotator/ckpts/body_pose_model.pth', type=str)
    parser.add_argument('--prompt', default='Close-up photo of the cool man and beautiful woman in surprised expressions as they accidentally discover a mysterious island while on vacation by the sea, 35mm photograph, film, professional, 4k, highly detailed.', type=str)
    parser.add_argument('--negative_prompt', default='noisy, blurry, soft, deformed, ugly', type=str)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--segment_type', default='yoloworld', help='GroundingDINO or yoloworld', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    prompts = [args.prompt]*2
    prompts_tmp = copy.deepcopy(prompts)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(device, args.segment_type)