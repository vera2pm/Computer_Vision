from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe.load_lora_weights("../pokemon-lora")
pipe = pipe.to("mps")  # or "cuda"

image = pipe("a cute pokemon llama").images[0]
image.save("pokemon.png")