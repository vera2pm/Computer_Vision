"""
LoRA fine-tune of Stable Diffusion v1.5 on the Pokemon dataset.

This is the plain-Python equivalent of:
  accelerate launch train_text_to_image_lora.py --pretrained_model_name_or_path=... --dataset_name=lambdalabs/pokemon-blip-captions ...

It does NOT train a model from scratch. stable-diffusion-v1-5 loads fully
pretrained and frozen; only small LoRA adapter matrices injected into the
UNet's attention layers are trained.

Install dependencies first:
  pip install torch torchvision diffusers transformers peft datasets pillow

Run with:
  python finetune_lora_pokemon.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_NAME = "reach-vb/pokemon-blip-captions" # "lambdalabs/pokemon-blip-captions"
OUTPUT_DIR = "../pokemon-lora"

RESOLUTION = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-6
LORA_RANK = 4

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Stick to float32 everywhere. fp16 is unstable on MPS, and this keeps the
# script identical across Mac / Colab / a real GPU box.
DTYPE = torch.float32

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PokemonDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, resolution):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # maps pixel values to [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        pixel_values = self.transform(example["image"].convert("RGB"))
        input_ids = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def main():
    # -----------------------------------------------------------------
    # Load the pretrained pieces (frozen — this is the "finetune" part:
    # nothing here starts from random weights)
    # -----------------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    text_encoder.to(DEVICE, dtype=DTYPE)
    vae.to(DEVICE, dtype=DTYPE)
    unet.to(DEVICE, dtype=DTYPE)

    # Freeze everything, then attach LoRA adapters to the UNet's attention
    # projections. These adapter weights are the only thing that trains.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    raw_dataset = load_dataset(DATASET_NAME, split="train")
    train_dataset = PokemonDataset(raw_dataset, tokenizer, RESOLUTION)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    unet.train()
    global_step = 0
    print("Start training...")
    for epoch in tqdm(range(NUM_EPOCHS)):
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
            input_ids = batch["input_ids"].to(DEVICE)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float())

            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  avg_loss={avg_loss:.4f}  global_step={global_step}")

    # -----------------------------------------------------------------
    # Save just the LoRA adapter weights (a few MB, not a new model)
    # -----------------------------------------------------------------
    lora_state_dict = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=OUTPUT_DIR,
        unet_lora_layers=lora_state_dict,
    )
    print(f"Saved LoRA weights to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()