import os
import sys

# ========================================
# 0. LOCAL CACHE - MUST COME FIRST!
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "huggingface")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")


# ========================================
# CHECK AND INSTALL MISSING DEPENDENCIES
# ========================================
def check_and_install(package_name, import_name=None):
    """Checks whether a package is installed, and installs it if missing."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"{package_name} not found. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
            print(f"{package_name} installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return False


# tiktoken is required by the Gemma tokenizer
check_and_install("tiktoken")

# Only import the heavy libraries now
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login

# ========================================
# AUTHENTICATION
# ========================================
# Reads the token from the HF_TOKEN environment variable.
# Only needed for gated models.
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("HF_TOKEN not set - proceeding without explicit Hugging Face login "
          "(set it if you need access to gated models).")

# ========================================
# CONFIGURATION: choose model and dataset
# ========================================
# NOTE:
# - MedGemma 4B is MULTIMODAL and needs more VRAM
# - EmbeddingGemma is an ENCODER (not a decoder) and needs a different setup
# - For standard causal-LM inference, use: gemma3_1b, codegemma, or vaultgemma

MODEL_CHOICE = "gemma3_1b"      # options: "gemma3_1b", "codegemma", "medgemma", "vaultgemma", "embeddinggemma"
DATASET_CHOICE = "ag_news"      # options: "ag_news", "wikitext", "imdb"

# -----------------------------
# Gemma model configurations
# -----------------------------
MODEL_CONFIGS = {
    "gemma3_1b": {
        "name": "google/gemma-3-1b-pt",
        "description": "Gemma 3 1B PT - Lightweight text-only model (good fit for a RTX 4070)",
        "batch_size": 4
    },
    "codegemma": {
        "name": "google/codegemma-2b",
        "description": "CodeGemma 2B - Specialized for code completion and generation",
        "batch_size": 3  # slightly reduced for the 2B model
    },
    "medgemma": {
        "name": "google/medgemma-4b-it",
        "description": "MedGemma 4B IT - Medical domain model (requires more VRAM)",
        "batch_size": 2  # reduced for the multimodal 4B model
    },
    "vaultgemma": {
        "name": "google/vaultgemma-1b",
        "description": "VaultGemma 1B - Privacy-preserving model trained with DP",
        "batch_size": 4
    },
    "embeddinggemma": {
        "name": "google/embeddinggemma-300m",
        "description": "EmbeddingGemma 300M - Specialized for embeddings (NOT for causal LM)",
        "batch_size": 8  # smaller model, larger batch size
    }
}

# -----------------------------
# Dataset configurations (compatible with all models)
# -----------------------------
DATASET_CONFIGS = {
    "ag_news": {
        "name": "fancyzhx/ag_news",
        "subset": None,
        "text_column": "text",
        "remove_columns": ["label"],
        "description": "News classification dataset (120k samples)"
    },
    "wikitext": {
        "name": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "text_column": "text",
        "remove_columns": [],
        "description": "Wikipedia articles for language modeling"
    },
    "imdb": {
        "name": "stanfordnlp/imdb",
        "subset": None,
        "text_column": "text",
        "remove_columns": ["label"],
        "description": "Movie reviews (50k samples)"
    }
}

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# -----------------------------
# 2. Model + tokenizer
# -----------------------------
model_config = MODEL_CONFIGS[MODEL_CHOICE]
model_name = model_config["name"]
BATCH_SIZE = model_config["batch_size"]

print(f"\n{'='*60}")
print(f"Loading model: {model_name}")
print(f"   Description: {model_config['description']}")
print(f"Downloading to: {CACHE_DIR}")
print(f"{'='*60}\n")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

# Set pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

print(f"Model loaded: {model_name}")

# Determine which device the model actually ended up on
if torch.cuda.is_available():
    model_device = next(model.parameters()).device
    print(f"Model is on device: {model_device}")
else:
    model_device = device

# -----------------------------
# 3. Dataset
# -----------------------------
dataset_config = DATASET_CONFIGS[DATASET_CHOICE]
print(f"\nLoading dataset: {dataset_config['name']}" +
      (f" ({dataset_config['subset']})" if dataset_config['subset'] else ""))
print(f"   Description: {dataset_config['description']}")

try:
    if dataset_config['subset']:
        dataset = load_dataset(
            dataset_config['name'],
            dataset_config['subset'],
            split="test",
            cache_dir=os.path.join(CACHE_DIR, "datasets")
        )
    else:
        dataset = load_dataset(
            dataset_config['name'],
            split="test",
            cache_dir=os.path.join(CACHE_DIR, "datasets")
        )
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise


def tokenize_function(examples):
    text_column = dataset_config['text_column']

    # Filter out empty examples
    texts = [text for text in examples[text_column] if text.strip()]

    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256  # matches the sequence length used at training time
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


print("\nTokenizing dataset...")

# Columns to drop
columns_to_remove = [dataset_config['text_column']] + dataset_config['remove_columns']
columns_to_remove = list(set(columns_to_remove))  # deduplicate

dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=columns_to_remove
)

# Filter out empty examples
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

dataset.set_format("torch")
print(f"Tokenization complete. Test samples: {len(dataset)}")

# -----------------------------
# 4. DataLoader
# -----------------------------
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True if torch.cuda.is_available() else False,
    num_workers=0
)

# -----------------------------
# 5. Timed inference loop
# -----------------------------
DURATION_SECONDS = 180  # 3 minutes
start_time = time.time()
iters = 0

print(f"\n{'='*60}")
print(f"Starting inference loop")
print(f"   Model: {MODEL_CHOICE} ({model_name})")
print(f"   Dataset: {DATASET_CHOICE}")
print(f"   Duration: {DURATION_SECONDS}s ({DURATION_SECONDS//60} minutes)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Max sequence length: 256 tokens")
print(f"   Precision: FP16")
print(f"   Cache: {CACHE_DIR}")
print(f"{'='*60}\n")

with torch.no_grad():
    while time.time() - start_time < DURATION_SECONDS:
        for batch in loader:
            # Move the batch to the model's device
            batch = {k: v.to(model_device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            iters += 1

            # Print progress every 50 iterations
            if iters % 50 == 0:
                elapsed = time.time() - start_time
                throughput_current = (iters * BATCH_SIZE) / elapsed
                remaining = DURATION_SECONDS - elapsed

                loss_info = f" | Loss: {outputs.loss.item():.4f}" if hasattr(outputs, 'loss') and outputs.loss is not None else ""

                print(f"Iter: {iters} | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s | Throughput: {throughput_current:.2f} samples/s{loss_info}")

            # Check elapsed time
            if time.time() - start_time >= DURATION_SECONDS:
                break

# -----------------------------
# 6. Final statistics
# -----------------------------
elapsed_time = time.time() - start_time
samples_processed = iters * BATCH_SIZE
throughput = samples_processed / elapsed_time

print(f"\n{'='*60}")
print(f"Inference complete!")
print(f"   Model: {MODEL_CHOICE} ({model_name})")
print(f"   Dataset: {DATASET_CHOICE}")
print(f"   Total iterations: {iters}")
print(f"   Total samples: {samples_processed}")
print(f"   Elapsed time: {elapsed_time:.2f}s")
print(f"   Average throughput: {throughput:.2f} samples/sec")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Sequence length: 256 tokens")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Max VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"{'='*60}")

print(f"\nAvailable models to try:")
for key, config in MODEL_CONFIGS.items():
    print(f"   - {key}: {config['description']}")
print(f"\nAvailable datasets:")
for key, config in DATASET_CONFIGS.items():
    print(f"   - {key}: {config['description']}")
