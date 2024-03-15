<div align="center">
<h1>OMG: Occlusion-friendly Personalized Multi-concept Generation In Diffusion Models</h1>
</div>

[![arXiv](https://img.shields.io/badge/ArXiv-2403-brightgreen)]()
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)]()
[![demo](https://img.shields.io/badge/Demo-Hugging%20Face-brightgreen)]()


<p align="center">
  <img src="assets/teaser.png">
</p>


[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/96vi3WFXTe0/1.jpg)](https://youtu.be/96vi3WFXTe0)
> **TL; DR:**  OMG is a framework for multi-concept image generation, supporting character and style LoRAs on [Civitai.com](https://civitai.com/). It also can be combined with [InstantID](https://github.com/InstantID/InstantID) for multiple IDs with using a single image for each ID.    

>  **Abstract:** *Personalization is an important topic in text-to-image generation, especially the challenging multi-concept personalization. Current multi-concept methods are struggling with identity preservation, occlusion, and the harmony between foreground and background. In this work, we propose OMG, an occlusion-friendly personalized generation framework designed to seamlessly integrate multiple concepts within a single image. We propose a novel two-stage sampling solution. The first stage takes charge of layout generation and visual comprehension information collection for handling occlusions. The second one utilizes the acquired visual comprehension information and the designed noise blending to integrate multiple concepts while considering occlusions. We also observe that the initiation denoising timestep for noise blending is the key to identity preservation and layout. Moreover, our method can be combined with various single-concept models, such as LoRA and InstantID without additional tuning. Especially, LoRA models on [civitai.com](https://civitai.com/) can be exploited directly. Extensive experiments demonstrate that OMG exhibits superior performance in multi-concept personalization.*


## 📝 Changelog

- __[2024.03.18]__: Release OMG code.
<br>

## :wrench: Dependencies and Installation


1. The code requires `python==3.10.6`, as well as `pytorch==2.0.1` and `torchvision==0.15.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```bash
conda create -n OMG python=3.10.6
conda activate OMG
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

2. For Visual comprehension, you can choose `YoloWorld + EfficientViT SAM` or `GroundingDINO + SAM`

- 1) (Recommend) YoloWorld + EfficientViT SAM:

```bash

pip install inference[yolo-world]==0.9.13
pip install  onnxsim==0.4.35

```

- 2) (Optional) If you can not install `inference[yolo-world]`. You can install `GroundingDINO` for visual comprehension.

`GroundingDINO` requires manual installation. 

Run this so the environment variable will be set under current shell.

```bash

export CUDA_HOME=/path/to/cuda-11.3

```

In this example, `/path/to/cuda-11.3` should be replaced with the path where your CUDA toolkit is installed.

```bash

git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO/

pip install -e .

```

More installation details can be found in [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#install)

## ⏬ Pretrained Model Preparation

Download [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
[InstantID](https://huggingface.co/InstantX/InstantID/tree/main), 
[antelopev2](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing),
[ControlNet](https://huggingface.co/lllyasviel/ControlNet),
[controlnet-openpose-sdxl-1.0](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0),
[controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0),
[controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0),
[dpt-hybrid-midas](https://huggingface.co/Intel/dpt-hybrid-midas).

For `YoloWorld + EfficientViT SAM`:
[EfficientViT-SAM-XL1](https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md), [yolo-world](https://huggingface.co/Fucius/OMG/blob/main/yolo-world.pt).

For `GroundingDINO + SAM`:
[GroundingDINO](https://huggingface.co/ShilongLiu/GroundingDINO), [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

For `Character LoRAs`:
[Civitai-Chris Evans](https://civitai.com/models/253793?modelVersionId=286084),
[Civitai-Taylor Swift](https://civitai.com/models/164284/taylor-swift?modelVersionId=185041),
[Harry Potter](https://huggingface.co/Fucius/OMG/blob/main/lora/Harry_Potter.safetensors),
[Hermione Granger](https://huggingface.co/Fucius/OMG/blob/main/lora/Hermione_Granger.safetensors).

For `Style LoRAs`:
[Anime Sketch Style](https://civitai.com/models/202764/anime-sketch-style-sdxl-and-sd15?modelVersionId=258108).

And put them under `checkpoint` as follow:
```angular2html
OMG
├── checkpoint
│   ├── antelopev2
│   ├── ControlNet
│   ├── controlnet-openpose-sdxl-1.0
│   ├── controlnet-canny-sdxl-1.0
│   ├── controlnet-depth-sdxl-1.0
│   ├── dpt-hybrid-midas
│   ├── style
│   ├── InstantID
│   ├── GroundingDINO
│   ├── lora
│   │   ├── Harry_Potter.safetensors
│   │   └── Hermione_Granger.safetensors
│   ├── sam
│   │   ├── sam_vit_h_4b8939.pth
│   │   └── xl1.pt
│   └── stable-diffusion-xl-base-1.0
├── gradio_demo
├── src
├── inference_instantid.py
└── inference_lora.py
```

If you use `YoloWorld`, put `yolo-world.pt` to `/tmp/cache/yolo_world/l/yolo-world.pt`. And put `ViT-B-32.pt` (download from [openai]( https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt )) to `~/.cache/clip/ViT-B-32.pt`

Or you can manually set the checkpoint path as follows:

For OMG + LoRA:
```
python inference_lora.py  \
--pretrained_sdxl_model <path to stable-diffusion-xl-base-1.0> \
--controlnet_checkpoint <path to controlnet-openpose-sdxl-1.0> \
--dino_checkpoint <path to GroundingDINO> \
--sam_checkpoint <path to sam> \
--lora_path <Lora path for character1|Lora path for character1>
```
For OMG + InstantID:
```
python inference_instantid.py  \
--pretrained_model <path to stable-diffusion-xl-base-1.0> \
--controlnet_path <path to InstantID controlnet> \
--face_adapter_path <path to InstantID face adapter>
--dino_checkpoint <path to GroundingDINO> \
--sam_checkpoint <path to sam> \
--antelopev2_path <path to antelopev2>
```

## :computer: Usage

### 1: OMG with LoRA
The &lt;TOK&gt; for `Harry_Potter.safetensors` is `Harry Potter` and for `Hermione_Granger.safetensors` is `Hermione Granger`.
```
python inference_lora.py \
    --prompt <prompt for the two person> \
    --negative_prompt <negative prompt> \
    --prompt_rewrite "[<prompt for person 1>]-*-[<negative prompt>]|[<prompt for person 2>]-*-[negative prompt]" \
    --lora_path "[<Lora path for character1|Lora path for character1>]"
```
For example:
```
python inference_lora.py \
    --prompt "Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed." \
    --negative_prompt 'noisy, blurry, soft, deformed, ugly' \
    --prompt_rewrite '[Close-up photo of the Harry Potter in surprised expressions as he wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]|[Close-up photo of the Hermione Granger in surprised expressions as she wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]' \
    --lora_path './checkpoint/lora/chris-evans.safetensors|./checkpoint/lora/TaylorSwiftSDXL.safetensors'
```
### 2: OMG with InstantID

```
python inference_instantid.py \
    --prompt <prompt for the two person> \
    --negative_prompt <negative prompt> \
    --prompt_rewrite "[<prompt for person 1>]-*-[<negative prompt>]-*-<path to reference image1>|[<prompt for person 2>]-*-[negative prompt]-*-<path to reference image2>",
```
For example:
```
python inference_instantid.py \
    --prompt 'Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed.' \
    --negative_prompt 'noisy, blurry, soft, deformed, ugly' \
    --prompt_rewrite '[Close-up photo of the a man, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]-*-./example/musk_resize.jpeg|[Close-up photo of the a man, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]-*-./example/yann-lecun_resize.jpg'
```

### 3. Local gradio demo with OMG + LoRA
If you choose `YoloWorld + EfficientViT SAM`:
```
python gradio_demo/app.py --segment_type yoloworld
```
For `GroundingDINO + SAM`:
```
python gradio_demo/app.py --segment_type GroundingDINO
```
Connect to the public URL displayed after the startup process is completed.
