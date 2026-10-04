import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

# ========================================
# CACHE & AUTH (optional)
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "huggingface")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.makedirs(CACHE_DIR, exist_ok=True)

# Reads the token from the HF_TOKEN environment variable.
# Only needed for gated/private models.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not set - proceeding without explicit Hugging Face login "
          "(set it if you need access to gated models).")

# ========================================
# CONFIGURATION
# ========================================
MODEL_CHOICE = "sd15"  # options: sd15, sdxl_turbo, tiny_sd
DATASET_CHOICE = "sdxl_creatures"  # options: naruto, sdxl_creatures

MODEL_CONFIGS = {
    "sd15": {
        "name": "runwayml/stable-diffusion-v1-5",
        "unet_subfolder": "unet",
        "vae_subfolder": "vae",
        "text_encoder": "clip",
        "resolution": 256,
        "use_ema": True,
        "recommended": True
    },
    "sdxl_turbo": {
        "name": "stabilityai/sdxl-turbo",
        "unet_subfolder": "unet",
        "vae_subfolder": "vae",
        "text_encoder": "dual_clip",  # SDXL has two text encoders
        "resolution": 512,
        "use_ema": False,  # turbo models typically don't use EMA
        "recommended": True
    },
    "tiny_sd": {
        "name": "segmind/tiny-sd",
        "unet_subfolder": "unet",
        "vae_subfolder": "vae",
        "text_encoder": "clip",
        "resolution": 256,  # the tiny model typically runs at 256
        "use_ema": False,
        "recommended": True
    }
}

DATASET_CONFIGS = {
    "sdxl_creatures": {
        "name": "Falah/sdxl-0.9",
        "image_column": "images",
        "caption_column": "Prompt",
        "remove_columns": ["model_name", "seed"]  # extra columns to drop
    },
    "naruto": {
        "name": "lambdalabs/naruto-blip-captions",
        "image_column": "image",
        "caption_column": "text",
        "remove_columns": None
    }
}

model_config = MODEL_CONFIGS[MODEL_CHOICE]
dataset_config = DATASET_CONFIGS[DATASET_CHOICE]

print(f"\n{'='*60}")
print(f"CONFIG: {MODEL_CHOICE} + {DATASET_CHOICE}")
print(f"   Model: {model_config['name']}")
print(f"   Resolution: {model_config['resolution']}")
print(f"{'='*60}\n")

# ========================================
# DEVICE
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ========================================
# LOAD MODEL
# ========================================
model_name = model_config["name"]
print(f"\nLoading {model_name}...")

# Load the main components
try:
    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder=model_config["unet_subfolder"],
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # VAE
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder=model_config["vae_subfolder"],
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # Tokenizer and text encoder
    if model_config["text_encoder"] == "clip":
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", cache_dir=CACHE_DIR
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    elif model_config["text_encoder"] == "dual_clip":
        # SDXL needs both tokenizers and text encoders
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        tokenizer_one = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", cache_dir=CACHE_DIR
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer_2", cache_dir=CACHE_DIR
        )
        text_encoder_one = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder_2", cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        # Keep them in a list for simplicity
        tokenizer = [tokenizer_one, tokenizer_two]
        text_encoder = [text_encoder_one, text_encoder_two]
    else:
        raise ValueError(f"Text encoder type {model_config['text_encoder']} not supported")

    # Noise scheduler (for training)
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name, subfolder="scheduler", cache_dir=CACHE_DIR
    )

    print("Components loaded")

except Exception as e:
    print(f"Error while loading: {e}")
    raise

# Move to device and set requires_grad
vae.requires_grad_(False)
vae.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
if isinstance(text_encoder, list):
    for te in text_encoder:
        te.requires_grad_(False)
        te.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
else:
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
unet.to(device, dtype=torch.float32)  # keep LoRA UNet in float32 for stability

# ========================================
# LORA ON THE UNET (optionally on the text encoder too)
# ========================================
# Enable gradient checkpointing to save memory
unet.enable_gradient_checkpointing()

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out", "proj_in"],
    lora_dropout=0.1,
    bias="none",
)
unet = get_peft_model(unet, lora_config)

# Optional: also apply LoRA to the text encoder (SD1.5-style models only)
if model_config["text_encoder"] == "clip" and False:  # set to True to also train the text encoder
    text_encoder_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder.print_trainable_parameters()

unet.print_trainable_parameters()
trainable_unet = sum(p.numel() for p in unet.parameters() if p.requires_grad)
total_unet = sum(p.numel() for p in unet.parameters())
print(f"LoRA on UNet: {trainable_unet:,} params ({100*trainable_unet/total_unet:.2f}%)")

# ========================================
# DATASET AND PREPROCESSING
# ========================================
print(f"\nLoading {dataset_config['name']}...")
try:
    if "subset" in dataset_config and dataset_config["subset"]:
        dataset = load_dataset(
            dataset_config["name"],
            dataset_config["subset"],
            cache_dir=os.path.join(CACHE_DIR, "datasets"),
            split="train",
        )
    else:
        dataset = load_dataset(
            dataset_config["name"],
            cache_dir=os.path.join(CACHE_DIR, "datasets"),
            split="train",
        )
    print(f"Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Image transforms
resolution = model_config["resolution"]
image_transforms = transforms.Compose([
    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # normalize to [-1, 1]
])


def preprocess_function(examples):
    images = []
    captions = []
    for img, cap in zip(examples[dataset_config["image_column"]], examples[dataset_config["caption_column"]]):
        if img is None or not cap:
            continue
        # img is usually a PIL.Image
        if isinstance(img, Image.Image):
            images.append(image_transforms(img))
        else:
            # If it's a path, load it (HF datasets usually already give PIL images)
            try:
                pil_img = Image.open(img).convert("RGB")
                images.append(image_transforms(pil_img))
            except Exception:
                continue
        captions.append(cap)
    return {"pixel_values": images, "captions": captions}


print("Preprocessing dataset...")
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    batch_size=32,  # batch the preprocessing for speed
)
dataset.set_format("torch")
print(f"Preprocessed: {len(dataset)} valid samples")

# DataLoader
train_dataloader = DataLoader(
    dataset,
    batch_size=4,  # adjust based on available VRAM
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# ========================================
# OPTIMIZER AND LR SCHEDULER
# ========================================
optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-8,
)

# Number of epochs
num_epochs = 100
num_update_steps_per_epoch = len(train_dataloader)
max_train_steps = num_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=max_train_steps,
)

# ========================================
# TRAINING LOOP
# ========================================
print(f"\n{'='*60}")
print(f"STARTING TRAINING")
print(f"   Epochs: {num_epochs}")
print(f"   Batch size: {train_dataloader.batch_size}")
print(f"   Total steps: {max_train_steps}")
print(f"{'='*60}\n")

global_step = 0
for epoch in range(num_epochs):
    unet.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(train_dataloader):
        # Extract pixel_values and captions
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        captions = batch["captions"]

        # Encode images to latents with the VAE
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            # Make sure latents match the UNet's dtype (e.g. if the VAE runs in fp16)
            if latents.dtype != unet.dtype:
                latents = latents.to(unet.dtype)

        # Sample a random timestep
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        # Add noise to the latents according to the scheduler
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embeddings
        with torch.no_grad():
            if isinstance(tokenizer, list):
                # SDXL: two tokenizers and text encoders
                text_inputs_one = tokenizer[0](
                    captions, padding="max_length", max_length=tokenizer[0].model_max_length, truncation=True, return_tensors="pt"
                )
                text_inputs_two = tokenizer[1](
                    captions, padding="max_length", max_length=tokenizer[1].model_max_length, truncation=True, return_tensors="pt"
                )
                encoder_hidden_states_one = text_encoder[0](
                    text_inputs_one.input_ids.to(device)
                )[0]
                encoder_hidden_states_two = text_encoder[1](
                    text_inputs_two.input_ids.to(device)
                )[0]
                # Concatenate the two representations (standard for SDXL)
                encoder_hidden_states = torch.cat([encoder_hidden_states_one, encoder_hidden_states_two], dim=-1)
            else:
                text_inputs = tokenizer(
                    captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

        encoder_hidden_states = encoder_hidden_states.to(unet.dtype)
        # Predict the noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Compute the loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        global_step += 1

        # Periodic logging
        if global_step % 100 == 0:
            print(f"Step {global_step}: loss = {loss.item():.6f}")

    progress_bar.close()

# ========================================
# SAVE LORA WEIGHTS
# ========================================
output_dir = f"./{MODEL_CHOICE}-{DATASET_CHOICE}-lora"
os.makedirs(output_dir, exist_ok=True)
unet.save_pretrained(output_dir)
print(f"\nLoRA weights saved to: {output_dir}")
