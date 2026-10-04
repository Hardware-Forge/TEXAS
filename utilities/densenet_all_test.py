import os

# ========================================
# 0. LOCAL CACHE
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "torch")

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# ========================================
# CONFIGURATION: choose model and dataset
# ========================================

MODEL_CHOICE = "densenet201"     # options: densenet121, densenet169, densenet201, densenet161
DATASET_CHOICE = "cifar10"       # options: cifar10, cifar100, svhn

# -----------------------------
# DenseNet model configurations
# -----------------------------
MODEL_CONFIGS = {
    "densenet121": {
        "name": "DenseNet-121",
        "model_fn": models.densenet121,
        "description": "DenseNet-121 - 8M parameters, 121 layers",
        "params": "~8M",
        "depth": 121,
        "growth_rate": 32,
        "vram_inference": "~1-2 GB",
        "recommended_batch": 128,
        "use_cases": [
            "Balanced image classification",
            "Efficient transfer learning",
            "Optimal feature reuse",
            "Good speed/accuracy trade-off"
        ]
    },
    "densenet169": {
        "name": "DenseNet-169",
        "model_fn": models.densenet169,
        "description": "DenseNet-169 - 14M parameters, 169 layers",
        "params": "~14M",
        "depth": 169,
        "growth_rate": 32,
        "vram_inference": "~2-3 GB",
        "recommended_batch": 128,
        "use_cases": [
            "Higher accuracy than DenseNet-121",
            "Complex classification tasks",
            "Fine-grained recognition",
            "Medical imaging"
        ]
    },
    "densenet201": {
        "name": "DenseNet-201",
        "model_fn": models.densenet201,
        "description": "DenseNet-201 - 20M parameters, 201 layers",
        "params": "~20M",
        "depth": 201,
        "growth_rate": 32,
        "vram_inference": "~2-3 GB",
        "recommended_batch": 128,
        "use_cases": [
            "Maximum accuracy",
            "Challenging datasets",
            "Research-grade classification",
            "High-quality feature extraction"
        ]
    },
    "densenet161": {
        "name": "DenseNet-161",
        "model_fn": models.densenet161,
        "description": "DenseNet-161 - 29M parameters, 161 layers (growth_rate=48)",
        "params": "~29M",
        "depth": 161,
        "growth_rate": 48,
        "vram_inference": "~3-4 GB",
        "recommended_batch": 128,
        "use_cases": [
            "Maximum representational capacity",
            "Large-scale classification",
            "Complex visual patterns",
            "State-of-the-art accuracy"
        ]
    }
}

# -----------------------------
# Dataset configurations
# -----------------------------
DATASET_CONFIGS = {
    "cifar10": {
        "name": "CIFAR-10",
        "dataset_class": CIFAR10,
        "num_classes": 10,
        "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "image_size": 32,
        "test_samples": 10000,
        "description": "10 common object classes",
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616]
    },
    "cifar100": {
        "name": "CIFAR-100",
        "dataset_class": CIFAR100,
        "num_classes": 100,
        "image_size": 32,
        "test_samples": 10000,
        "description": "100 fine-grained classes",
        "mean": [0.5071, 0.4867, 0.4408],
        "std": [0.2675, 0.2565, 0.2761]
    },
    "svhn": {
        "name": "SVHN (Street View House Numbers)",
        "dataset_class": SVHN,
        "num_classes": 10,
        "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        "image_size": 32,
        "test_samples": 26032,
        "description": "Digit recognition from Google Street View",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970]
    }
}

# -----------------------------
# Configuration validation
# -----------------------------
model_config = MODEL_CONFIGS[MODEL_CHOICE]
dataset_config = DATASET_CONFIGS[DATASET_CHOICE]

print(f"\n{'='*70}")
print(f"INFERENCE CONFIGURATION")
print(f"{'='*70}")
print(f"Model: {MODEL_CHOICE}")
print(f"   Name: {model_config['name']}")
print(f"   Parameters: {model_config['params']}")
print(f"   Depth: {model_config['depth']} layers")
print(f"   Growth rate: {model_config['growth_rate']}")
print(f"   VRAM inference: {model_config['vram_inference']}")
print(f"   Recommended batch: {model_config['recommended_batch']}")
print(f"\nUse cases:")
for use_case in model_config['use_cases']:
    print(f"   - {use_case}")
print(f"\nDataset: {dataset_config['name']}")
print(f"   {dataset_config['description']}")
print(f"   Test samples: {dataset_config['test_samples']:,}")
print(f"   Num classes: {dataset_config['num_classes']}")
print(f"{'='*70}\n")

# -----------------------------
# 1. Device and VRAM check
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Total VRAM: {total_vram:.1f} GB")
    torch.cuda.empty_cache()
    available_vram = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"   Available VRAM: {available_vram:.1f} GB\n")

# -----------------------------
# 2. Data transforms
# -----------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
])

# -----------------------------
# 3. Dataset loading
# -----------------------------
print(f"Loading {dataset_config['name']} test set...")

if DATASET_CHOICE == "svhn":
    testset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        split='test',
        download=True,
        transform=transform_test
    )
else:
    testset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        train=False,
        download=True,
        transform=transform_test
    )

BATCH_SIZE = model_config['recommended_batch']

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"Dataset loaded")
print(f"   Test samples: {len(testset):,}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Num batches: {len(testloader)}")

# -----------------------------
# 4. Model
# -----------------------------
print(f"\nBuilding {model_config['name']}...")

# Load pretrained model
model = model_config['model_fn'](weights='DEFAULT')

# Replace classifier head for this dataset
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, dataset_config['num_classes'])

model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded")
print(f"   Total parameters: {total_params:,}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    used_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"   VRAM used: {used_vram:.2f} GB")

# -----------------------------
# 5. Timed inference loop
# -----------------------------
DURATION_SECONDS = 300  # 5 minutes
criterion = nn.CrossEntropyLoss()

print(f"\n{'='*60}")
print(f"Starting inference benchmark")
print(f"   Model: {model_config['name']}")
print(f"   Dataset: {dataset_config['name']}")
print(f"   Duration: {DURATION_SECONDS}s ({DURATION_SECONDS//60} minutes)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")

print(f"Starting inference...\n")

start_time = time.time()
iters = 0
total_loss = 0.0
correct = 0
total_samples = 0

with torch.no_grad():
    while time.time() - start_time < DURATION_SECONDS:
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            iters += 1

            # Log every 50 iterations
            if iters % 50 == 0:
                elapsed = time.time() - start_time
                remaining = DURATION_SECONDS - elapsed
                throughput = total_samples / elapsed
                avg_loss = total_loss / iters
                accuracy = 100. * correct / total_samples

                vram_info = ""
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / 1024**3
                    vram_info = f" | VRAM: {vram_used:.2f}GB"

                print(f"Iter: {iters:4d} | Elapsed: {elapsed:5.1f}s | "
                      f"Remaining: {remaining:5.1f}s | Throughput: {throughput:7.1f} img/s | "
                      f"Acc: {accuracy:.2f}% | Loss: {avg_loss:.4f}{vram_info}")

            # Check time
            if time.time() - start_time >= DURATION_SECONDS:
                break

        if time.time() - start_time >= DURATION_SECONDS:
            break

# -----------------------------
# 6. Final statistics
# -----------------------------
elapsed_time = time.time() - start_time
throughput = total_samples / elapsed_time
avg_loss = total_loss / iters
accuracy = 100. * correct / total_samples

print(f"\n{'='*70}")
print(f"INFERENCE COMPLETE")
print(f"{'='*70}")
print(f"Model: {model_config['name']}")
print(f"   Parameters: {model_config['params']}")
print(f"   Depth: {model_config['depth']} layers")
print(f"\nPerformance metrics:")
print(f"   Total iterations: {iters}")
print(f"   Total images processed: {total_samples:,}")
print(f"   Elapsed time: {elapsed_time:.2f}s ({elapsed_time/60:.1f} min)")
print(f"   Average throughput: {throughput:.1f} images/sec")
print(f"   Average loss: {avg_loss:.4f}")
print(f"   Accuracy: {accuracy:.2f}%")
print(f"\nConfiguration:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Dataset: {dataset_config['name']}")
print(f"   Image size: {dataset_config['image_size']}x{dataset_config['image_size']}")

if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nGPU stats:")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Peak VRAM: {max_vram:.2f} GB")
    print(f"   VRAM efficiency: {(max_vram / total_vram) * 100:.1f}%")

print(f"{'='*70}")
