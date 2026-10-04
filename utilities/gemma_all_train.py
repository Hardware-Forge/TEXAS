import os

# ========================================
# 0. LOCAL CACHE - MUST COME FIRST!
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "huggingface")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")

# Only import transformers / huggingface_hub now
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import torch
from peft import LoraConfig, get_peft_model, TaskType

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

MODEL_CHOICE = "gemma3_1b"      # options: "gemma3_1b", "codegemma", "medgemma", "vaultgemma", "embeddinggemma"
DATASET_CHOICE = "ag_news"      # options: "ag_news", "wikitext", "imdb"

# -----------------------------
# Gemma model configurations
# -----------------------------
MODEL_CONFIGS = {
    "gemma3_1b": {
        "name": "google/gemma-3-1b-pt",  # pretrained version
        "description": "Gemma 3 1B PT - Lightweight text-only model (good fit for a RTX 4070)"
    },
    "codegemma": {
        "name": "google/codegemma-2b",  # 2B base model for code
        "description": "CodeGemma 2B - Specialized for code completion and generation"
    },
    "medgemma": {
        "name": "google/medgemma-4b-it",  # 4B instruction-tuned (smallest available)
        "description": "MedGemma 4B IT - Medical domain model (requires more VRAM)"
    },
    "vaultgemma": {
        "name": "google/vaultgemma-1b",  # 1B trained with differential privacy
        "description": "VaultGemma 1B - Privacy-preserving model trained with DP"
    },
    "embeddinggemma": {
        "name": "google/embeddinggemma-300m",  # 300M embedding model
        "description": "EmbeddingGemma 300M - Specialized for text embeddings (smallest model)"
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
print(f"\n{'='*60}")
print(f"Loading model: {model_name}")
print(f"   Description: {model_config['description']}")
print(f"Downloading to: {CACHE_DIR}")
print(f"{'='*60}\n")

# Load without quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16 to save memory
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

# Set pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print(f"Model loaded: {model_name}")

# ========================================
# LORA SETUP (reduces memory and training time)
# ========================================
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"LoRA applied - training {trainable_params:,} params ({100 * trainable_params / total_params:.2f}%)")

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
            cache_dir=os.path.join(CACHE_DIR, "datasets")
        )
    else:
        dataset = load_dataset(
            dataset_config['name'],
            cache_dir=os.path.join(CACHE_DIR, "datasets")
        )
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Drop unneeded columns
if dataset_config['remove_columns']:
    dataset = dataset.remove_columns(dataset_config['remove_columns'])

print(f"Dataset loaded: {dataset_config['name']}")
print(f"   Train samples: {len(dataset['train'])}")

# -----------------------------
# 4. Tokenization
# -----------------------------
def tokenize_function(examples):
    text_column = dataset_config['text_column']

    # Filter out empty examples (important for wikitext)
    texts = [text for text in examples[text_column] if text.strip()]

    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256  # increased from 128 since we're not quantizing
    )

    # For causal language modeling, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


print("\nTokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[dataset_config['text_column']]
)

# Filter out empty examples
tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)

tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"]
print(f"Tokenization complete. Training samples: {len(train_dataset)}")

# -----------------------------
# 5. TrainingArguments
# -----------------------------
output_dir = f"./{MODEL_CHOICE}-{DATASET_CHOICE}-lora"

# Adjust batch size based on the model
# MedGemma 4B needs more memory than the other 1-2B models
if MODEL_CHOICE == "medgemma":
    per_device_batch_size = 2  # reduced for MedGemma 4B
    gradient_accumulation = 8   # increased to compensate
    print("MedGemma 4B detected - using a reduced batch size (2) for a RTX 4070")
else:
    per_device_batch_size = 4  # normal batch size for 1-2B models
    gradient_accumulation = 4

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    num_train_epochs=2,              # 2 epochs gives decent results
    logging_steps=50,
    save_steps=1000,
    fp16=True,                       # FP16 to optimize memory usage
    learning_rate=2e-4,              # learning rate for LoRA
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=100,                # warmup for stability
    max_grad_norm=1.0,
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# -----------------------------
# 7. Start training
# -----------------------------
print(f"\n{'='*60}")
print(f"Starting training with LoRA")
print(f"   Model: {MODEL_CHOICE} ({model_name})")
print(f"   Dataset: {DATASET_CHOICE}")
print(f"   Output: {output_dir}")
print(f"   Cache: {CACHE_DIR}")
print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   Effective batch size: {per_device_batch_size * gradient_accumulation}")
print(f"   Max sequence length: 256 tokens")
print(f"   Precision: FP16")
print(f"{'='*60}\n")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()

print(f"\nTraining complete! LoRA adapters saved to: {output_dir}")
print(f"\nTo use the model:")
print(f"   from peft import PeftModel")
print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
print(f"   ")
print(f"   model = AutoModelForCausalLM.from_pretrained('{model_name}', torch_dtype=torch.float16)")
print(f"   model = PeftModel.from_pretrained(model, '{output_dir}')")
print(f"   tokenizer = AutoTokenizer.from_pretrained('{model_name}')")
print(f"\nAvailable models to try:")
for key, config in MODEL_CONFIGS.items():
    print(f"   - {key}: {config['description']}")
print(f"\nAvailable datasets:")
for key, config in DATASET_CONFIGS.items():
    print(f"   - {key}: {config['description']}")
