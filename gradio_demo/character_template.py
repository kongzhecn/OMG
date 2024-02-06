character_list_man = [
    {
        "name": "Harry Potter",
        "prompt": "Close-up photo of the Harry Potter, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Harry_Potter.safetensors",
    },
]

character_list_woman = [
    {
        "name": "Hermione Granger",
        "prompt": "Close-up photo of the Hermione Granger, 35mm photograph, film, professional, 4k, highly detailed.",
        "negative_prompt": "noisy, blurry, soft, deformed, ugly",
        "path": "./checkpoint/lora/Hermione_Granger.safetensors",
    },
]

character_man = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_man}
character_woman = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in character_list_woman}

lorapath_man = {k["name"]: (k["path"]) for k in character_list_man}
lorapath_woman = {k["name"]: (k["path"]) for k in character_list_woman}