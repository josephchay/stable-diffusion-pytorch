# Stable Diffusion with PyTorch

Pytorch implementation of Stable Diffusion with DDPM pipeline

---

# Weights & Tokenizers
1. Download `vocab.json` and `merges.txt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in the `assets` folder
2. Download `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the `assets` folder

# Inferencing
1. Set the paths to the tokenizers and the model weights files in the `sd.ipynb`.
```python
VOCAB_FILE_PATH = "../assets/vocab.json"
MERGES_FILE_PATH = "../assets/merges.txt"
MODEL_FILE_PATH = "../assets/v1-5-pruned-emaonly.ckpt"
```

2. Set the configuration hyperparameters.
```markdown
- `prompt` (str): The text prompt to be used for generating the output image.
- `uncond_prompt` (str): Also known as negative prompt. Default: `""`
- `do_cfg` (bool): Default: `True`
- `cfg_scale` (int): Default: `8`. min: 1, max: 14
- `sampler` (str): Default: `"ddpm"`
- `num_inference_steps` (int): Default: `50`
- `seed` (int): Default: `42`
- `strength` (float): Higher values add more noise to input image and will make the output image less similar to it.
- `input_image` (PIL.Image): The input image to be used for generating the output image. Not need for TTI Default: `None`
- `idle_device` (str): Default: `"cpu"`
- `device` (str): The device to be used for generating the output image.
- `models` (dict): The models to be used for generating the output image.
- `tokenizer` (transformers.CLIPTokenizer): The tokenizer to be used for generating the output image.
```

3. Create sample prompts
```python
prompts = [
    "A futuristic cityscape with neon lights and flying cars, detailed cyberpunk aesthetic, ultra sharp, 35mm lens, 8k resolution.",
    "A medieval castle on a hilltop at sunset, with dramatic lighting and clouds in the sky, high contrast, ultra sharp, 24mm lens, 8k resolution.",
    "A cozy cabin in the woods with a warm fire burning inside, snow falling outside, soft focus, 50mm lens, 8k resolution.",
    ...
]
```

## Image generation
### TTI (Text to Image)
1. Set up the configuration for tti.
```python
config_tti = {
    "uncond_prompt": uncond_prompt,
    "do_cfg": do_cfg,
    "cfg_scale": cfg_scale,
    "strength": .9,  # from range of 0.0 to 1.0
    "sampler_name": sampler,
    "n_inference_steps": num_inference_steps,
    "seed": seed
}
```

2. Generate the image with a specific text prompt
```python
output_image_tti = generate_img(prompts[9], None, config_tti)
Image.fromarray(output_image_tti)
```

### ITI (Image to Image with Text Prompt)
1. Set up the configuration for iti.
```python
config_iti = {
    "uncond_prompt": uncond_prompt,
    "do_cfg": do_cfg,
    "cfg_scale": cfg_scale,
    "strength": .6,  # from range of 0.0 to 1.0
    "sampler_name": sampler,
    "n_inference_steps": num_inference_steps,
    "seed": seed
}
```

2. Generate the image with a specific text prompt and input image
```python
output_image_iti = generate_img(night_city_prompts[3], Image.open("../assets/images/nightcity.jpg"), config_iti)
# Combine the input image and the output image into a single image.
Image.fromarray(output_image_iti)
```
