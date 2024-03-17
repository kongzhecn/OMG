character_list_man = [
    {
        "name": "Harry Potter (identifier: Harry Potter)",
        "prompt": "Close-up photo of the Harry Potter, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Harry_Potter.safetensors",
    },
    {
        "name": "Chris Evans (identifier: Chris Evans)",
        "prompt": "Close-up photo of the Chris Evans, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/chris-evans.safetensors",
    },
    {
        "name": "Jordan Torres (identifier: jordan_torres)",
        "prompt": "Close-up photo of the jordan_torres man, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/jordan_torres_v2_xl.safetensors",
    },
]

character_list_woman = [
    {
        "name": "Hermione Granger (identifier: Hermione Granger)",
        "prompt": "Close-up photo of the Hermione Granger, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Hermione_Granger.safetensors",
    },
    {
        "name": "Taylor Swift (identifier: TaylorSwift)",
        "prompt": "Close-up photo of the TaylorSwift, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/TaylorSwiftSDXL.safetensors",
    },
    {
        "name": "Keira Knightley (identifier: ohwx woman)",
        "prompt": "Close-up photo of the ohwx woman, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/keira_lora_sdxl_v1-000008.safetensors",
    },
]

style_list = [
    {
        "name": "None",
        "prompt": "",
        "path": "",
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
    },
    {
        "name": "Cinematic Photography Style",
        "prompt": "Cinematic Hollywood Film Style, ",
        "path": "./checkpoint/style/Cinematic Hollywood Film.safetensors",
    }
]

character_man = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_man}
character_woman = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_woman}
styles = {k["name"]: (k["prompt"]) for k in style_list}

lorapath_man = {k["name"]: (k["path"]) for k in character_list_man}
lorapath_woman = {k["name"]: (k["path"]) for k in character_list_woman}
lorapath_styles = {k["name"]: (k["path"]) for k in style_list}