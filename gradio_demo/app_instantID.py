import sys
sys.path.append('./')
import argparse
import hashlib
import json
import os.path
import numpy as np
import torch
from typing import Tuple, List
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import T2IAdapter
from PIL import Image
import copy
from diffusers import ControlNetModel, StableDiffusionXLPipeline
from insightface.app import FaceAnalysis
import gradio as gr
import random
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

from src.pipelines.instantid_pipeline import InstantidMultiConceptPipeline
from src.pipelines.instantid_single_pieline import InstantidSingleConceptPipeline
from src.prompt_attention.p2p_attention import AttentionReplace
from src.pipelines.instantid_pipeline import revise_regionally_controlnet_forward
import cv2
import math
import PIL.Image

from gradio_demo.character_template import styles, lorapath_styles
STYLE_NAMES = list(styles.keys())



MAX_SEED = np.iinfo(np.int32).max

title = r"""
<h1 align="center"> OMG + InstantID </h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/kongzhecn/OMG/' target='_blank'><b>OMG: Occlusion-friendly Personalized Multi-concept Generation In Diffusion Models</b></a>.<be><br>
<br>
<a href='https://kongzhecn.github.io/omg-project/' target='_blank'><b>[Project]</b></a><a href='https://github.com/kongzhecn/OMG/' target='_blank'><b>[Code]</b></a><a href='https://arxiv.org/abs/2403.10983/' target='_blank'><b>[Arxiv]</b></a><br>
<br>
❗️<b>Related demos<b>:<a href='https://huggingface.co/spaces/Fucius/OMG/' target='_blank'><b> OMG + LoRAs</b></a>❗️<br>
<br>
How to use:<br>
1. Input two images for a man and a woman.
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
@article{kong2024omg,
  title={OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models},
  author={Kong, Zhe and Zhang, Yong and Yang, Tianyu and Wang, Tao and Zhang, Kaihao and Wu, Bizhu and Chen, Guanying and Liu, Wei and Luo, Wenhan},
  journal={arXiv preprint arXiv:2403.10983},
  year={2024}
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



def build_dino_segment_model(ckpt_repo_id, sam_checkpoint):
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return groundingdino_model, sam_predictor

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    args = SLConfig.fromfile(ckpt_config_filename)
    model = build_model(args)
    args.device = device

    checkpoint = torch.load(os.path.join(repo_id, filename), map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(filename, log))
    _ = model.eval()
    return model

def build_yolo_segment_model(sam_path, device):
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url=sam_path).to(device).eval()
    )
    return yolo_world, sam

def sample_image(pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    concept_models=None,
    num_inference_steps=50,
    guidance_scale=3.0,
    controller=None,
    face_app=None,
    image=None,
    stage=None,
    region_masks=None,
    controlnet_conditioning_scale=None,
    **extra_kargs
):

    if image is not None:
        image_condition = [image]
    else:
        image_condition = None


    images = pipe(
        prompt=input_prompt,
        concept_models=concept_models,
        negative_prompt=input_neg_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": 0.8},
        controller=controller,
        image=image_condition,
        face_app=face_app,
        stage=stage,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        region_masks=region_masks,
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

def draw_kps_multi(image_pil, kps_list, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])


    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for kps in kps_list:
        kps = np.array(kps)
        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

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

def build_model_sd(pretrained_model, controlnet_path, face_adapter, device, prompts, antelopev2_path, width, height, style_lora):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16).to(device)
    pipe = InstantidMultiConceptPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to(device)

    controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.},
                                  self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device, width=width, height=height,
                                  dtype=torch.float16)
    revise_regionally_controlnet_forward(pipe.unet, controller)

    controlnet_concept = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe_concept = InstantidSingleConceptPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet_concept,
        torch_dtype=torch.float16
    )
    pipe_concept.load_ip_adapter_instantid(face_adapter)
    pipe_concept.set_ip_adapter_scale(0.8)
    pipe_concept.to(device)
    pipe_concept.image_proj_model.to(pipe_concept._execution_device)

    if style_lora is not None and os.path.exists(style_lora):
        pipe.load_lora_weights(style_lora, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')
        pipe_concept.load_lora_weights(style_lora, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')


    # modify
    app = FaceAnalysis(name='antelopev2', root=antelopev2_path,
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    return pipe, controller, pipe_concept, app


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
        prompt_region, neg_prompt_region, ref_img = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')

        region_collection.append((prompt_region, neg_prompt_region, ref_img))
    return (prompt, region_collection)

def build_model_lora(pipe, pipe_concept, style_path, condition, condition_img):
    if condition == "Human pose" and condition_img is not None:
        controlnet = ControlNetModel.from_pretrained(args.openpose_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.controlnet2 = controlnet
    elif condition == "Canny Edge" and condition_img is not None:
        controlnet = ControlNetModel.from_pretrained(args.canny_checkpoint, torch_dtype=torch.float16, variant="fp16").to(device)
        pipe.controlnet2 = controlnet
    elif condition == "Depth" and condition_img is not None:
        controlnet = ControlNetModel.from_pretrained(args.depth_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.controlnet2 = controlnet

    if style_path is not None and os.path.exists(style_path):
        pipe_concept.load_lora_weights(style_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')
        pipe.load_lora_weights(style_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='style')

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

def get_example():
    case = [
        [
            'close-up shot, photography, a man and a woman on the street, facing the camera smiling',
            '../example/schmidhuber_resize.png',
            '../example/Hermione Granger.jpg',
            'None',
            'None',
            None
        ],
        [
            'close-up shot, photography, a man and a woman on the street, facing the camera smiling',
            '../example/schmidhuber_resize.png',
            '../example/Hermione Granger.jpg',
            'None',
            'None',
            None
        ]
    ]
    return case



def main(device, segment_type):
    pipe, controller, pipe_concepts, face_app = build_model_sd(args.pretrained_model, args.controlnet_path,
                                                               args.face_adapter_path, device, prompts_tmp,
                                                               args.antelopev2_path, width // 32, height // 32,
                                                               args.style_lora)
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

    prompts_rewrite = [args.prompt_rewrite]
    input_prompt_test = [prepare_text(p, p_w) for p, p_w in zip(prompts, prompts_rewrite)]
    input_prompt_test = [prompts, input_prompt_test[0][1]]

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

    def get_depth(image, height, weight):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, weight),
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


    def generate_image(prompt1, negative_prompt, reference_1, reference_2, resolution, local_prompt1, local_prompt2, seed, style, identitynet_strength_ratio, adapter_strength_ratio, condition, condition_img, controlnet_ratio, cfg_scale):
        identitynet_strength_ratio = float(identitynet_strength_ratio)
        adapter_strength_ratio = float(adapter_strength_ratio)
        controlnet_ratio = float(controlnet_ratio)
        cfg_scale = float(cfg_scale)

        if lorapath_styles[style] is not None and os.path.exists(lorapath_styles[style]):
            styleL = True
        else:
            styleL = False

        width, height = int(resolution.split("*")[0]), int(resolution.split("*")[1])
        kwargs = {
            'height': height,
            'width': width,
            't2i_controlnet_conditioning_scale': controlnet_ratio,
        }

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
            spatial_condition = get_depth(condition_img, height, width)
        else:
            spatial_condition = None

        kwargs['t2i_image'] = spatial_condition
        pipe.unload_lora_weights()
        pipe_concepts.unload_lora_weights()
        build_model_lora(pipe, pipe_concepts, lorapath_styles[style], condition, condition_img)
        pipe_concepts.set_ip_adapter_scale(adapter_strength_ratio)

        input_list = [prompt1]


        for prompt in input_list:
            if prompt != '':
                input_prompt = []
                p = '{prompt}, 35mm photograph, film, professional, 4k, highly detailed.'
                if styleL:
                    p = styles[style] + p
                input_prompt.append([p.replace('{prompt}', prompt), p.replace("{prompt}", prompt)])
                if styleL:
                    input_prompt.append([(styles[style] + local_prompt1, 'noisy, blurry, soft, deformed, ugly',
                                          PIL.Image.fromarray(reference_1)),
                                         (styles[style] + local_prompt2, 'noisy, blurry, soft, deformed, ugly',
                                          PIL.Image.fromarray(reference_2))])
                else:
                    input_prompt.append(
                        [(local_prompt1, 'noisy, blurry, soft, deformed, ugly', PIL.Image.fromarray(reference_1)),
                         (local_prompt2, 'noisy, blurry, soft, deformed, ugly', PIL.Image.fromarray(reference_2))])


                controller.reset()
                image = sample_image(
                    pipe,
                    input_prompt=input_prompt,
                    concept_models=pipe_concepts,
                    input_neg_prompt=[negative_prompt] * len(input_prompt),
                    generator=torch.Generator(device).manual_seed(seed),
                    controller=controller,
                    face_app=face_app,
                    controlnet_conditioning_scale=identitynet_strength_ratio,
                    stage=1,
                    guidance_scale=cfg_scale,
                    **kwargs)

                controller.reset()

                if pipe.tokenizer("man")["input_ids"][1] in pipe.tokenizer(args.prompt)["input_ids"][1:-1]:
                    mask1 = predict_mask(detect_model, sam, image[0], 'man', args.segment_type, confidence=0.1,
                                         threshold=0.5)
                else:
                    mask1 = None

                if pipe.tokenizer("woman")["input_ids"][1] in pipe.tokenizer(args.prompt)["input_ids"][1:-1]:
                    mask2 = predict_mask(detect_model, sam, image[0], 'woman', args.segment_type, confidence=0.1,
                                         threshold=0.5)
                else:
                    mask2 = None

                if mask1 is not None or mask2 is not None:
                    face_info = face_app.get(cv2.cvtColor(np.array(image[0]), cv2.COLOR_RGB2BGR))
                    face_kps = draw_kps_multi(image[0], [face['kps'] for face in face_info])

                    image = sample_image(
                        pipe,
                        input_prompt=input_prompt,
                        concept_models=pipe_concepts,
                        input_neg_prompt=[negative_prompt] * len(input_prompt),
                        generator=torch.Generator(device).manual_seed(seed),
                        controller=controller,
                        face_app=face_app,
                        image=face_kps,
                        stage=2,
                        controlnet_conditioning_scale=identitynet_strength_ratio,
                        region_masks=[mask1, mask2],
                        guidance_scale=cfg_scale,
                        **kwargs)

                # return [image[1], spatial_condition]
                return image

    with gr.Blocks(css=css) as demo:
        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            gallery = gr.Image(label="Generated Images", height=512, width=512)
            gallery1 = gr.Image(label="Generated Images", height=512, width=512)
            usage_tips = gr.Markdown(label="Usage tips of OMG", value=tips, visible=False)


        with gr.Row():
            reference_1 = gr.Image(label="Input an RGB image for Character man", height=128, width=128)
            reference_2 = gr.Image(label="Input an RGB image for Character woman", height=128, width=128)
            condition_img1 = gr.Image(label="Input an RGB image for condition (Optional)", height=128, width=128)




        with gr.Row():
            local_prompt1 = gr.Textbox(label="Character1_prompt",
                                info="Describe the Character 1",
                                value="Close-up photo of the a man, 35mm photograph, professional, 4k, highly detailed.")
            local_prompt2 = gr.Textbox(label="Character2_prompt",
                                       info="Describe the Character 2",
                                       value="Close-up photo of the a woman, 35mm photograph, professional, 4k, highly detailed.")
        with gr.Row():
            identitynet_strength_ratio = gr.Slider(
                label="IdentityNet strength (for fidelity)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            adapter_strength_ratio = gr.Slider(
                label="Image adapter strength (for detail)",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=0.80,
            )
            controlnet_ratio = gr.Slider(
                label="ControlNet strength",
                minimum=0,
                maximum=1.5,
                step=0.05,
                value=1,
            )

            cfg_ratio = gr.Slider(
                label="CFG scale ",
                minimum=0.5,
                maximum=10,
                step=0.5,
                value=3.0,
            )

            resolution = gr.Dropdown(label="Image Resolution (width*height)", choices=resolution_list,
                                     value="1024*1024")
            style = gr.Dropdown(label="style", choices=STYLE_NAMES, value="None")
            condition = gr.Dropdown(label="Input condition type", choices=condition_list, value="None")


        # prompt
        with gr.Column():
            prompt = gr.Textbox(label="Prompt",
                                info="Give a simple prompt to describe the image content",
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
            inputs=[prompt, negative_prompt, reference_1, reference_2, resolution, local_prompt1, local_prompt2, seed, style, identitynet_strength_ratio, adapter_strength_ratio, condition, condition_img1, controlnet_ratio, cfg_ratio],
            outputs=[gallery, gallery1]
        )


        gr.Markdown(article)
    demo.launch(server_name='0.0.0.0',server_port=7861, debug=True)

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='./checkpoint/YamerMIX_v8', type=str)
    parser.add_argument('--controlnet_path', default='./checkpoint/InstantID/ControlNetModel', type=str)
    parser.add_argument('--face_adapter_path', default='./checkpoint/InstantID/ip-adapter.bin', type=str)
    parser.add_argument('--openpose_checkpoint', default='./checkpoint/controlnet-openpose-sdxl-1.0', type=str)
    parser.add_argument('--canny_checkpoint', default='./checkpoint/controlnet-canny-sdxl-1.0', type=str)
    parser.add_argument('--depth_checkpoint', default='./checkpoint/controlnet-depth-sdxl-1.0', type=str)
    parser.add_argument('--dpt_checkpoint', default='./checkpoint/dpt-hybrid-midas', type=str)
    parser.add_argument('--pose_detector_checkpoint',
                        default='./checkpoint/ControlNet/annotator/ckpts/body_pose_model.pth', type=str)
    parser.add_argument('--efficientViT_checkpoint', default='./checkpoint/sam/xl1.pt', type=str)
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--antelopev2_path', default='./checkpoint/antelopev2', type=str)
    parser.add_argument('--save_dir', default='results/instantID', type=str)
    parser.add_argument('--prompt', default='Close-up photo of the cool man and beautiful woman as they accidentally discover a mysterious island while on vacation by the sea, facing the camera smiling, 35mm photograph, film, professional, 4k, highly detailed.', type=str)
    parser.add_argument('--negative_prompt', default='noisy, blurry, soft, deformed, ugly', type=str)
    parser.add_argument('--prompt_rewrite',
                        default='[Close-up photo of a man, 35mm photograph, professional, 4k, highly detailed.]-*'
                                '-[noisy, blurry, soft, deformed, ugly]-*-'
                                '../example/chris-evans.jpg|'
                                '[Close-up photo of a woman, 35mm photograph, professional, 4k, highly detailed.]-'
                                '*-[noisy, blurry, soft, deformed, ugly]-*-'
                                '../example/TaylorSwift.png',
                        type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--segment_type', default='yoloworld', help='GroundingDINO or yoloworld', type=str)
    parser.add_argument('--style_lora', default='', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    prompts = [args.prompt] * 2

    prompts_tmp = copy.deepcopy(prompts)

    width, height = 1024, 1024
    kwargs = {
        'height': height,
        'width': width,
    }

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(device, args.segment_type)

