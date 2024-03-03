<div align="center">
<h1>MMC: Mastering Multiple Concepts in One Frame</h1>
</div>

## :label: TODO 

- [x] Release inference code and demo.
- [x] Release checkpoints.


## :wrench: Dependencies and Installation


The code requires `python==3.10.6`, as well as `pytorch==2.0.1` and `torchvision==0.15.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```bash
conda create -n MMC python=3.10.6
conda activate MMC
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

[//]: # (2.`GroundingDINO` requires manual installation. )

[//]: # ()
[//]: # (Run this so the environment variable will be set under current shell.)

[//]: # (```bash)

[//]: # (export CUDA_HOME=/path/to/cuda-11.3)

[//]: # (```)

[//]: # (In this example, `/path/to/cuda-11.3` should be replaced with the path where your CUDA toolkit is installed.)

[//]: # (```bash)

[//]: # (git clone https://github.com/IDEA-Research/GroundingDINO.git)

[//]: # (cd GroundingDINO/)

[//]: # (pip install -e .)

[//]: # (```)

[//]: # (More installation details can be found in [GroundingDINO]&#40;https://github.com/IDEA-Research/GroundingDINO#install&#41;)

## ⏬ Pretrained Model Preparation

Download [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [EfficientViT-SAM-XL1](https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md), [lora](https://huggingface.co/Fucius/MMC_Lora), [controlnet-openpose-sdxl-1.0](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0), [InstantID](https://huggingface.co/InstantX/InstantID/tree/main), [antelopev2](https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing) and put them under `checkpoint` as follow:
```angular2html
MMC
├── checkpoint
│   ├── antelopev2
│   ├── controlnet-openpose-sdxl-1.0
│   ├── InstantID
│   ├── lora
│   │   ├── Harry_Potter.safetensors
│   │   └── Hermione_Granger.safetensors
│   ├── sam
│   │   └── xl1.pt
│   └── stable-diffusion-xl-base-1.0
├── gradio_demo
├── src
├── inference_instantid.py
└── inference_lora.py
```
Or you can manually set the checkpoint path as follows:

For MMC + LoRA:
```
python inference_lora.py  \
--pretrained_sdxl_model <path to stable-diffusion-xl-base-1.0> \
--controlnet_checkpoint <path to controlnet-openpose-sdxl-1.0> \
--dino_checkpoint <path to GroundingDINO> \
--sam_checkpoint <path to sam> \
--lora_path <path to loar1|path to lora2>
```
For MMC + InstantID:
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

### 1: MMC with LoRA
The &lt;TOK&gt; for `Harry_Potter.safetensors` is `Harry Potter` and for `Hermione_Granger.safetensors` is `Hermione Granger`.
```
python inference_lora.py \
    --prompt <prompt for the two person> \
    --negative_prompt <negative prompt> \
    --prompt_rewrite "[<prompt for person 1>]-*-[<negative prompt>]|[<prompt for person 2>]-*-[negative prompt]",
```
For example:
```
python inference_lora.py \
    --prompt "Close-up photo of the happy smiles on the faces of the cool man and beautiful woman as they leave the island with the treasure, sail back to the vacation beach, and begin their love story, 35mm photograph, film, professional, 4k, highly detailed." \
    --negative_prompt 'noisy, blurry, soft, deformed, ugly' \
    --prompt_rewrite '[Close-up photo of the Harry Potter in surprised expressions as he wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]|[Close-up photo of the Hermione Granger in surprised expressions as she wear Hogwarts uniform, 35mm photograph, film, professional, 4k, highly detailed.]-*-[noisy, blurry, soft, deformed, ugly]'
```
### 2: MMC with InstantID

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

### 3. Local gradio demo with MMC + LoRA
```
python gradio_demo/app.py 
```
Connect to the public URL displayed after the startup process is completed.