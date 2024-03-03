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
    guidance_scale=7.5,
    controller=None,
    face_app=None,
    image=None,
    stage=None,
    region_masks=None,
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
            sam.set_image(image_source, image_format="RGB")
            masks, _, _ = sam.predict(box=detections.xyxy[0], multimask_output=False)
            masks = torch.from_numpy(masks.squeeze())

    return masks

def build_model_sd(pretrained_model, controlnet_path, face_adapter, device, prompts, antelopev2_path):
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = InstantidMultiConceptPipeline.from_pretrained(
        pretrained_model, controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to(device)

    controller = AttentionReplace(prompts, 50, cross_replace_steps={"default_": 1.},
                                  self_replace_steps=0.4, tokenizer=pipe.tokenizer, device=device,
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

    # modify
    app = FaceAnalysis(name='antelopev2', root=antelopev2_path,
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='./checkpoint/stable-diffusion-xl-base-1.0', type=str)
    parser.add_argument('--controlnet_path', default='./checkpoint/InstantID/ControlNetModel', type=str)
    parser.add_argument('--face_adapter_path', default='./checkpoint/InstantID/ip-adapter.bin', type=str)
    parser.add_argument('--efficientViT_checkpoint', default='./checkpoint/sam/xl1.pt', type=str)
    parser.add_argument('--dino_checkpoint', default='./checkpoint/GroundingDINO', type=str)
    parser.add_argument('--sam_checkpoint', default='./checkpoint/sam/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument('--antelopev2_path', default='./checkpoint/antelopev2', type=str)
    parser.add_argument('--save_dir', default='results/multiInstantID', type=str)
    parser.add_argument('--prompt', default='Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed.', type=str)
    parser.add_argument('--negative_prompt', default='noisy, blurry, soft, deformed, ugly', type=str)
    parser.add_argument('--prompt_rewrite',
                        default='[Close-up photo of the a man, 35mm photograph, film, professional, 4k, highly detailed.]-*'
                                '-[noisy, blurry, soft, deformed, ugly]-*-'
                                './example/musk_resize.jpeg|'
                                '[Close-up photo of the a woman, 35mm photograph, film, professional, 4k, highly detailed.]-'
                                '*-[noisy, blurry, soft, deformed, ugly]-*-'
                                './example/Johansson.jpg',
                        type=str)
    parser.add_argument('--seed', default=30, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--segment_type', default='yoloworld', help='GroundingDINO or yoloworld', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prompts = [args.prompt] * 2

    prompts_tmp = copy.deepcopy(prompts)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe, controller, pipe_concepts, face_app = build_model_sd(args.pretrained_model, args.controlnet_path, args.face_adapter_path, device, prompts_tmp, args.antelopev2_path)
    if args.segment_type == 'GroundingDINO':
        detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)
    else:
        detect_model, sam = build_yolo_segment_model(args.efficientViT_checkpoint, device)

    width, height = 1024, 1024

    kwargs = {
        'height': height,
        'width': width,
    }

    # prompts = [args.prompt]
    prompts_rewrite = [args.prompt_rewrite]
    input_prompt = [prepare_text(p, p_w) for p, p_w in zip(prompts, prompts_rewrite)]
    input_prompt = [prompts, input_prompt[0][1]]
    save_prompt = input_prompt[0][0]

    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        concept_models=pipe_concepts,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        controller=controller,
        face_app=face_app,
        stage=1,
        **kwargs)

    controller.reset()

    mask1 = predict_mask(detect_model, sam, image[0], 'man', args.segment_type, confidence=0.2, threshold=0.5)
    mask2 = predict_mask(detect_model, sam, image[0], 'woman', args.segment_type, confidence=0.2, threshold=0.5)

    face_info = face_app.get(cv2.cvtColor(np.array(image[0]), cv2.COLOR_RGB2BGR))
    face_kps = draw_kps_multi(image[0], [face['kps'] for face in face_info])

    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        concept_models=pipe_concepts,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        controller=controller,
        face_app=face_app,
        image = face_kps,
        stage=2,
        region_masks=[mask1, mask2],
        **kwargs)


    configs = [
        f'pretrained_model: {args.pretrained_model}\n',
        f'context_prompt: {args.prompt}\n', f'neg_context_prompt: {args.negative_prompt}\n',
        f'prompt_rewrite: {args.prompt_rewrite}\n'
    ]
    hash_code = hashlib.sha256(''.join(configs).encode('utf-8')).hexdigest()[:8]

    save_name = f'**---{args.suffix}---{hash_code}.png'
    save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    save_path = os.path.join(save_dir, save_name)
    save_config_path = os.path.join(save_dir, save_name.replace('.png', '.txt'))
    attention_path = os.path.join(save_dir, save_name).replace('**', 'att')

    print(f'save to: {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    stage1_path = os.path.join(save_dir, 'stage-1.png')
    stage2_path = os.path.join(save_dir, 'stage-2.png')
    image[0].save(stage1_path)
    image[1].save(stage2_path)

    with open(save_config_path, 'w') as fw:
        fw.writelines(configs)
