import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline  # for SDXL
from huggingface_hub import login

# ========================================
# CACHE & AUTH
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "huggingface")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.makedirs(CACHE_DIR, exist_ok=True)


def check_install(pkg, imp=None):
    imp = imp or pkg
    try:
        __import__(imp)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])


check_install("tiktoken")

# Reads the token from the HF_TOKEN environment variable.
# Only needed for gated models.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not set - proceeding without explicit Hugging Face login "
          "(set it if you need access to gated models).")

# ========================================
# CONFIGURATION
# ========================================
MODEL_CHOICE = "sd15"             # options: sd15, sdxl_turbo, tiny_sd
DATASET_CHOICE = "sdxl_creatures"  # options: sdxl_creatures, naruto
LORA_PATH = None                  # path to trained LoRA weights (e.g. "./sd15-tuxemon-lora")
# LORA_PATH = "./sd15-tuxemon-lora"  # example

MODEL_CONFIGS = {
    "sd15": {
        "name": "runwayml/stable-diffusion-v1-5",
        "pipeline_class": StableDiffusionPipeline,
        "resolution": 512,
        "batch_size": 4,          # adjust based on available VRAM
        "text_encoder": "clip",
    },
    "sdxl_turbo": {
        "name": "stabilityai/sdxl-turbo",
        "pipeline_class": StableDiffusionXLPipeline,  # SDXL needs its own pipeline class
        "resolution": 512,
        "batch_size": 2,
        "text_encoder": "dual_clip",
    },
    "tiny_sd": {
        "name": "segmind/tiny-sd",
        "pipeline_class": StableDiffusionPipeline,
        "resolution": 256,
        "batch_size": 4,
        "text_encoder": "clip",
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
print(f"   Batch size: {model_config['batch_size']}")
if LORA_PATH:
    print(f"   LoRA weights: {LORA_PATH}")
print(f"{'='*60}\n")

# ========================================
# DEVICE
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {total_vram:.1f} GB")
    torch.cuda.empty_cache()

# ========================================
# LOAD PIPELINE
# ========================================
model_name = model_config["name"]
BATCH_SIZE = model_config["batch_size"]

print(f"\nLoading pipeline {model_name}...")

try:
    # Load base pipeline
    pipeline = model_config["pipeline_class"].from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir=CACHE_DIR,
        safety_checker=None,          # disable safety checker for speed
        requires_safety_checker=False,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)  # no progress bar in the loop

    # If a LoRA path is given, load and merge the weights
    if LORA_PATH and os.path.exists(LORA_PATH):
        print(f"   Loading LoRA weights from {LORA_PATH}")
        # Assumes the weights were saved with unet.save_pretrained()
        from peft import PeftModel
        # The pipeline holds a UNet, so we can wrap it with PeftModel
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, LORA_PATH)
        # Optional: merge the weights for faster inference
        pipeline.unet = pipeline.unet.merge_and_unload()
        print("   LoRA applied and merged.")

    print("Pipeline loaded")
    if torch.cuda.is_available():
        print(f"   VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

except Exception as e:
    print(f"Error: {e}")
    raise

# ========================================
# DATASET (prompts only)
# ========================================
print(f"\nLoading {dataset_config['name']}...")
try:
    if "subset" in dataset_config and dataset_config["subset"]:
        dataset = load_dataset(
            dataset_config["name"],
            dataset_config["subset"],
            split=dataset_config.get("split", "train"),
            cache_dir=os.path.join(CACHE_DIR, "datasets"),
        )
    else:
        dataset = load_dataset(
            dataset_config["name"],
            split=dataset_config.get("split", "train"),
            cache_dir=os.path.join(CACHE_DIR, "datasets"),
        )
    print(f"Dataset: {len(dataset)} prompts")
except Exception as e:
    print(f"Error: {e}")
    raise


# Keep only the prompt column
def extract_prompts(examples):
    return {"prompt": examples[dataset_config["caption_column"]]}


dataset = dataset.map(extract_prompts, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type="torch", columns=["prompt"])  # not a tensor, but this is fine

# DataLoader iterating over prompts
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda x: {"prompt": [item["prompt"] for item in x]}  # handles the strings
)

# ========================================
# INFERENCE LOOP
# ========================================
DURATION = 300  # 5 minutes
start = time.time()
iters = 0
total_images = 0

# Directory for a few example outputs
example_dir = os.path.join(PROJECT_DIR, "inference_examples")
os.makedirs(example_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"INFERENCE - {DURATION//60} minutes")
print(f"   Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")

with torch.no_grad():
    while time.time() - start < DURATION:
        for batch in loader:
            prompts = batch["prompt"]
            batch_start = time.time()

            # Generate images
            images = pipeline(
                prompt=prompts,
                num_inference_steps=30,          # adjust as needed
                guidance_scale=7.5,               # for standard models; use 0.0 for turbo models
                output_type="pil"
            ).images

            batch_time = time.time() - batch_start
            iters += 1
            total_images += len(images)

            # Save the first 5 batches as examples
            if iters <= 5:
                for i, img in enumerate(images):
                    img.save(os.path.join(example_dir, f"example_{iters}_{i}.png"))

            # Periodic logging
            if iters % 50 == 0:
                elapsed = time.time() - start
                throughput = total_images / elapsed
                remaining = DURATION - elapsed
                vram = ""
                if torch.cuda.is_available():
                    vram = f" | VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB"
                print(f"{iters:4d} | {elapsed:5.1f}s | -{remaining:5.1f}s | {throughput:6.2f} img/s{vram}")

            if time.time() - start >= DURATION:
                break

        if time.time() - start >= DURATION:
            break

# ========================================
# RESULTS
# ========================================
elapsed = time.time() - start
throughput = total_images / elapsed

print(f"\n{'='*60}")
print(f"COMPLETE")
print(f"{'='*60}")
print(f"Model: {MODEL_CHOICE}")
print(f"Iterations: {iters}")
print(f"Images generated: {total_images}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
print(f"Throughput: {throughput:.2f} images/sec")
print(f"Examples saved in: {example_dir}")

if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak VRAM: {max_vram:.2f} GB ({max_vram/total_vram*100:.1f}%)")

print(f"{'='*60}")
