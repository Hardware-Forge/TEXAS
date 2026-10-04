import os

# ========================================
# 0. LOCAL CACHE - MUST COME FIRST!
# ========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache", "torch")

os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Cache directory: {CACHE_DIR}")

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# ========================================
# CONFIGURATION: choose model and dataset
# ========================================

MODEL_CHOICE = "densenet201"     # options: densenet121, densenet169, densenet201, densenet161
DATASET_CHOICE = "svhn"          # options: cifar10, cifar100, svhn

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
        "vram_training": "~2-3 GB",
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
        "vram_training": "~3-4 GB",
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
        "vram_training": "~4-5 GB",
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
        "vram_training": "~5-6 GB",
        "recommended_batch": 12,
        "use_cases": [
            "Maximum representational capacity",
            "Large-scale classification",
            "Complex visual patterns",
            "State-of-the-art accuracy"
        ],
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
        "train_samples": 50000,
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
        "train_samples": 50000,
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
        "train_samples": 73257,
        "test_samples": 26032,
        "description": "Digit recognition from Google Street View",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
        "split_name": "train"  # SVHN uses 'train' instead of a boolean flag
    }
}

# -----------------------------
# Configuration validation
# -----------------------------
model_config = MODEL_CONFIGS[MODEL_CHOICE]
dataset_config = DATASET_CONFIGS[DATASET_CHOICE]

print(f"\n{'='*70}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*70}")
print(f"Model: {MODEL_CHOICE}")
print(f"   Name: {model_config['name']}")
print(f"   Parameters: {model_config['params']}")
print(f"   Depth: {model_config['depth']} layers")
print(f"   Growth rate: {model_config['growth_rate']}")
print(f"   VRAM training: {model_config['vram_training']}")
print(f"   Recommended batch: {model_config['recommended_batch']}")
print(f"\nUse cases:")
for use_case in model_config['use_cases']:
    print(f"   - {use_case}")
print(f"\nDataset: {dataset_config['name']}")
print(f"   {dataset_config['description']}")
print(f"   Training samples: {dataset_config['train_samples']:,}")
print(f"   Test samples: {dataset_config['test_samples']:,}")
print(f"   Num classes: {dataset_config['num_classes']}")
print(f"   Image size: {dataset_config['image_size']}x{dataset_config['image_size']}")
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
# 2. Data transforms and augmentation
# -----------------------------
print(f"Setting up data augmentation...")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dataset_config['mean'], dataset_config['std'])
])

print(f"Augmentation configured: RandomCrop + HorizontalFlip + Normalize")

# -----------------------------
# 3. Dataset loading
# -----------------------------
print(f"\nLoading {dataset_config['name']}...")

if DATASET_CHOICE == "svhn":
    # SVHN uses 'train'/'test' splits instead of train=True/False
    trainset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        split='train',
        download=True,
        transform=transform_train
    )
    testset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        split='test',
        download=True,
        transform=transform_test
    )
else:
    trainset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = dataset_config['dataset_class'](
        root=CACHE_DIR,
        train=False,
        download=True,
        transform=transform_test
    )

BATCH_SIZE = model_config['recommended_batch']

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE * 2,  # larger batch for evaluation
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"Dataset loaded")
print(f"   Train batches: {len(trainloader)}")
print(f"   Test batches: {len(testloader)}")
print(f"   Batch size (train): {BATCH_SIZE}")
print(f"   Batch size (test): {BATCH_SIZE * 2}")

# -----------------------------
# 4. Model
# -----------------------------
print(f"\nBuilding {model_config['name']}...")

# Load ImageNet-pretrained model
model = model_config['model_fn'](weights='DEFAULT')

# Replace classifier head for this dataset's class count
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, dataset_config['num_classes'])

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model built")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    used_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"   VRAM used (model only): {used_vram:.2f} GB")

# -----------------------------
# 5. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

print(f"\nOptimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)")
print(f"   Scheduler: CosineAnnealingLR (T_max=200)")
print(f"   Loss: CrossEntropyLoss")

# -----------------------------
# 6. Training loop
# -----------------------------
NUM_EPOCHS = 10  # lower for a quick test run, use 200 for better final results
output_dir = f"./{MODEL_CHOICE}_{DATASET_CHOICE}_trained"
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Starting training")
print(f"   Model: {model_config['name']}")
print(f"   Dataset: {dataset_config['name']}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Output dir: {output_dir}")
print(f"{'='*60}\n")

best_acc = 0.0
training_stats = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(trainloader)}] "
                  f"Loss: {loss.item():.3f} Acc: {100.*correct/total:.2f}%")

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total

    # Validation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total

    epoch_time = time.time() - epoch_start

    # Update scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Log epoch
    stats = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'lr': current_lr,
        'time': epoch_time
    }
    training_stats.append(stats)

    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] complete")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    print(f"   LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

    # Save best model
    if test_acc > best_acc:
        print(f"   New best accuracy: {test_acc:.2f}% (previous: {best_acc:.2f}%)")
        best_acc = test_acc
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'model_config': model_config,
            'dataset_config': dataset_config
        }
        torch.save(checkpoint, os.path.join(output_dir, 'best_model.pth'))
        print(f"   Checkpoint saved!")

    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# 7. Training complete
# -----------------------------
print(f"\n{'='*70}")
print(f"TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Model: {model_config['name']}")
print(f"Dataset: {dataset_config['name']}")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"Model saved to: {output_dir}/best_model.pth")

if torch.cuda.is_available():
    max_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nGPU stats:")
    print(f"   Peak VRAM: {max_vram:.2f} GB")

print(f"{'='*70}")
