character_list_man = [
    {
        "name": "Chris Evans (identifier: Chris Evans)",
        "prompt": "Close-up photo of the Chris Evans, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/chris-evans.safetensors",
    },
{
        "name": "Harry Potter (identifier: Harry Potter)",
        "prompt": "Close-up photo of the Harry Potter, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Harry_Potter.safetensors",
    },
    {
        "name": "Jordan Torres (identifier: jordan_torres)",
        "prompt": "Close-up photo of the jordan_torres man, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/jordan_torres_v2_xl.safetensors",
    },
    {
        "name": "Gleb Savchenko (identifier: Liam Hemsworth a man)",
        "prompt": "Close-up photo of Liam Hemsworth a man, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/gleb_savchenko_sdxl.safetensors",
    },
]

character_list_woman = [
    {
        "name": "Taylor Swift (identifier: TaylorSwift)",
        "prompt": "Close-up photo of the TaylorSwift, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/TaylorSwiftSDXL.safetensors",
    },
    {
        "name": "Hermione Granger (identifier: Hermione Granger)",
        "prompt": "Close-up photo of the Hermione Granger, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Hermione_Granger.safetensors",
    },
    {
        "name": "Keira Knightley (identifier: ohwx woman)",
        "prompt": "Close-up photo of the ohwx woman, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/keira_lora_sdxl_v1-000008.safetensors",
    },
    {
        "name": "Jennifer Lawrence (identifier: Jennifer Lawrence WOMAN)",
        "prompt": "Close-up photo of the Jennifer Lawrence WOMAN, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/lawrence_dh128_v1-step00012000.safetensors",
    },
]

style_list = [
    {
        "name": "None",
        "prompt": "",
        "path": "",
    },
    {
        "name": "Cinematic Photography Style",
        "prompt": "Cinematic Hollywood Film Style, ",
        "path": "./checkpoint/style/Cinematic Hollywood Film.safetensors",
    },
    {
        "name": "Anime sketch style",
        "prompt": "Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch, ",
        "path": "./checkpoint/style/Anime_Sketch_SDXL.safetensors",
    },
    {
        "name": "Oil Painting Style",
        "prompt": "palette knife painting, ",
        "path": "./checkpoint/style/EldritchPaletteKnife.safetensors",
    }
]

character_man = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_man}
character_woman = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_woman}
styles = {k["name"]: (k["prompt"]) for k in style_list}

lorapath_man = {k["name"]: (k["path"]) for k in character_list_man}
lorapath_woman = {k["name"]: (k["path"]) for k in character_list_woman}
lorapath_styles = {k["name"]: (k["path"]) for k in style_list}