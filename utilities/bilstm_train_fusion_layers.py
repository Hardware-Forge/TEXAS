import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

# -----------------------------
# MODEL CONFIGURATION
# -----------------------------
# Specify here the models to use and their number of convolutional layers.
MODEL_CONFIG = {
    # "densenet121": {"num_conv_layers": 120, "folders": ["densenet-121"]},
    "densenet169": {"num_conv_layers": 168, "folders": ["DenseNet-169"]},
    "densenet201": {"num_conv_layers": 200, "folders": ["DenseNet-201"]},
    "mobilenetv2": {"num_conv_layers": 53, "folders": ["MobileNet"]},
    "resnet18": {"num_conv_layers": 20, "folders": ["ResNet-18"]},
    "resnet34": {"num_conv_layers": 36, "folders": ["ResNet-34"]},
    "resnet101": {"num_conv_layers": 103, "folders": ["ResNet-101"]},
    "paligemma": {"num_conv_layers": 1, "folders": ["paligemma"]},
    "mining": {"num_conv_layers": 0, "folders": ["mining"]},
    # "gru": {"num_conv_layers": 0, "folders": ["gru"]},
}

# -----------------------------
# Parameters
# -----------------------------
DATA_DIRS = ["src/Attack/TEX", "src/Attack/Async"]
SEQ_LEN = 256
STRIDE = 16
BATCH_SIZE = 256
NUM_EPOCHS = 150
LR = 1e-3
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_PER_CSV = 300000

# -----------------------------
# Dual-Stream Dataset for Regression
# -----------------------------
class DualStreamLayerCountDataset(Dataset):
    """
    Loads pairs of sequences (TEX, Async) and attaches the number of
    convolutional layers as the regression target.
    """
    def __init__(self, csv_pairs_with_count, seq_len=256, stride=16, train=True,
                 split_ratio=0.8, max_sequences_per_csv=50000):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []
        self.labels = []

        for tex_csv, async_csv, model_name, num_conv in tqdm(csv_pairs_with_count,
                                                               desc="Loading CSV pairs",
                                                               unit="pair"):
            # Load TEX
            df_tex = pd.read_csv(tex_csv)
            latencies_tex = df_tex['latency_cycles'].values

            # Load Async
            df_async = pd.read_csv(async_csv)
            latencies_async = df_async['latency_cycles'].values

            # Check minimum length
            if len(latencies_tex) < seq_len or len(latencies_async) < seq_len:
                continue

            # Random block selection for TEX
            max_block_len_tex = min(len(latencies_tex), max_sequences_per_csv * stride + seq_len)
            start_row_tex = random.randint(0, len(latencies_tex) - max_block_len_tex)
            block_tex = latencies_tex[start_row_tex:start_row_tex + max_block_len_tex]

            # Random block selection for Async
            max_block_len_async = min(len(latencies_async), max_sequences_per_csv * stride + seq_len)
            start_row_async = random.randint(0, len(latencies_async) - max_block_len_async)
            block_async = latencies_async[start_row_async:start_row_async + max_block_len_async]

            # Train/val split
            split_idx_tex = int(len(block_tex) * split_ratio)
            split_idx_async = int(len(block_async) * split_ratio)

            if train:
                block_tex = block_tex[:split_idx_tex]
                block_async = block_async[:split_idx_async]
            else:
                block_tex = block_tex[split_idx_tex:]
                block_async = block_async[split_idx_async:]

            # Generate indices for the smaller number of sequences between the two streams
            num_seqs_tex = len(block_tex) - seq_len
            num_seqs_async = len(block_async) - seq_len
            num_seqs = min(num_seqs_tex, num_seqs_async)

            if num_seqs <= 0:
                continue

            indices = range(0, num_seqs, stride)
            indices = list(indices)[:max_sequences_per_csv]

            for i in indices:
                self.data.append((block_tex, i, block_async, i))
                self.labels.append(float(num_conv))  # label = number of conv layers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        block_tex, start_tex, block_async, start_async = self.data[idx]

        # Extract sequences
        seq_tex = block_tex[start_tex:start_tex + self.seq_len]
        seq_async = block_async[start_async:start_async + self.seq_len]

        # Normalize each stream independently
        seq_tex = (seq_tex - seq_tex.mean()) / (seq_tex.std() + 1e-8)
        seq_async = (seq_async - seq_async.mean()) / (seq_async.std() + 1e-8)

        return (torch.tensor(seq_tex, dtype=torch.float32),
                torch.tensor(seq_async, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

# -----------------------------
# Dual-Stream BiLSTM Regressor
# -----------------------------
class DualStreamBiLSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()

        self.lstm_tex = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True)
        self.lstm_async = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True)

        self.fc_fusion = nn.Linear(hidden_size * 4, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        # Single output for regression (number of conv layers)
        self.fc_out = nn.Linear(128, 1)

        # ReLU to keep the output non-negative
        self.relu = nn.ReLU()

    def forward(self, x_tex, x_async):
        x_tex = x_tex.unsqueeze(-1)
        x_async = x_async.unsqueeze(-1)

        out_tex, _ = self.lstm_tex(x_tex)
        feat_tex = out_tex[:, -1, :]

        out_async, _ = self.lstm_async(x_async)
        feat_async = out_async[:, -1, :]

        combined = torch.cat([feat_tex, feat_async], dim=-1)

        x = torch.relu(self.fc_fusion(combined))
        x = self.bn(x)
        x = self.dropout(x)
        out = self.fc_out(x)

        # ReLU to avoid negative layer counts
        out = self.relu(out)

        return out.squeeze(-1)  # output shape: (batch_size,)

# -----------------------------
# Helper to build CSV pairs with their layer count
# -----------------------------
def create_csv_pairs_by_layer_count(data_dirs, model_config):
    """
    Builds (TEX, Async) CSV pairs for each model specified in model_config.
    Pairs files by name: async_results_* with tex_results_*.
    USES EVERY FILE for training/validation (none held out).
    Returns: train_val_pairs
    """
    SEED = 42
    random.seed(SEED)

    train_val_pairs = []

    print("\n" + "="*60)
    print("CONVOLUTIONAL LAYER COUNT REGRESSION")
    print("="*60)
    print(f"\nConfigured models:")
    for model_name, config in MODEL_CONFIG.items():
        print(f"  {model_name}: {config['num_conv_layers']} conv layers")

    print("\n" + "="*60)
    print("Building CSV pairs (matched by file name)")
    print("="*60)

    for model_name, config in model_config.items():
        num_conv = config['num_conv_layers']
        folders = config['folders']

        print(f"\n{model_name} ({num_conv} conv layers):")

        # Find the CSVs for this model
        tex_csvs = []
        async_csvs = []

        for folder in folders:
            # Search in TEX
            for data_dir in data_dirs:
                if 'TEX' in data_dir:
                    pattern = os.path.join(data_dir, "**", "plots", folder, "*.csv")
                    found = glob.glob(pattern, recursive=True)
                    tex_csvs.extend(found)
                elif 'Async' in data_dir:
                    pattern = os.path.join(data_dir, "**", "plots", folder, "*.csv")
                    found = glob.glob(pattern, recursive=True)
                    async_csvs.extend(found)

        print(f"  TEX files found: {len(tex_csvs)}")
        print(f"  Async files found: {len(async_csvs)}")

        if not tex_csvs or not async_csvs:
            print(f"  Skipped (missing files in one of the two streams)")
            continue

        # Build a dict for pairing based on file name.
        # Key: the common part of the name (after tex_results_ or async_results_)
        tex_dict = {}
        for tex_file in tex_csvs:
            basename = os.path.basename(tex_file)
            if basename.startswith('tex_results_'):
                # Extract the part after 'tex_results_'
                key = basename.replace('tex_results_', '')
                tex_dict[key] = tex_file

        async_dict = {}
        for async_file in async_csvs:
            basename = os.path.basename(async_file)
            if basename.startswith('async_results_'):
                # Extract the part after 'async_results_'
                key = basename.replace('async_results_', '')
                async_dict[key] = async_file

        # Find the common keys (files present in both TEX and Async)
        common_keys = set(tex_dict.keys()) & set(async_dict.keys())

        print(f"  Valid pairs found: {len(common_keys)}")

        if not common_keys:
            print(f"  No matching pairs found!")
            print(f"  TEX key examples: {list(tex_dict.keys())[:3]}")
            print(f"  Async key examples: {list(async_dict.keys())[:3]}")
            continue

        # Build the pairs
        num_pairs = 0
        for key in common_keys:
            tex_file = tex_dict[key]
            async_file = async_dict[key]
            train_val_pairs.append((tex_file, async_file, model_name, num_conv))
            num_pairs += 1

        print(f"  -> {num_pairs} pairs created for train/val")

        # Show a few pairing examples
        if num_pairs > 0:
            print(f"  Pairing examples:")
            for i, key in enumerate(list(common_keys)[:3]):
                print(f"    {i+1}. {key}")
                print(f"       TEX:   {os.path.basename(tex_dict[key])}")
                print(f"       Async: {os.path.basename(async_dict[key])}")

    print(f"\nTotal train/val pairs: {len(train_val_pairs)}")

    # Final shuffle to mix models together
    random.shuffle(train_val_pairs)

    return train_val_pairs

# -----------------------------
# Regression metrics
# -----------------------------
def calculate_regression_metrics(predictions, targets):
    """
    Computes regression metrics.
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)

    # Exact match accuracy (rounded)
    rounded_preds = np.round(predictions)
    exact_matches = (rounded_preds == targets).sum()
    exact_accuracy = exact_matches / len(targets)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'exact_accuracy': exact_accuracy,
        'mape': mape
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()

    # Check the configuration
    if not MODEL_CONFIG:
        print("ERROR: MODEL_CONFIG is empty! Configure the models before running.")
        exit(1)

    # Build CSV pairs (uses every file)
    train_val_pairs = create_csv_pairs_by_layer_count(DATA_DIRS, MODEL_CONFIG)

    if not train_val_pairs:
        print("\nERROR: No file pairs found! Check:")
        print("  1. The paths in DATA_DIRS")
        print("  2. The folder names in MODEL_CONFIG")
        print("  3. That .csv files exist in the given folders")
        exit(1)

    # Save configuration
    with open("layer_count_config.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("LAYER COUNT CONFIGURATION\n")
        f.write("="*60 + "\n\n")
        for model_name, config in MODEL_CONFIG.items():
            f.write(f"{model_name}: {config['num_conv_layers']} convolutional layers\n")
            f.write(f"  Folders: {', '.join(config['folders'])}\n\n")

    # Datasets
    print("\n" + "="*60)
    print("Loading dataset")
    print("="*60)

    train_dataset = DualStreamLayerCountDataset(
        train_val_pairs, seq_len=SEQ_LEN, stride=STRIDE,
        train=True, max_sequences_per_csv=MAX_SEQ_PER_CSV
    )
    val_dataset = DualStreamLayerCountDataset(
        train_val_pairs, seq_len=SEQ_LEN, stride=STRIDE,
        train=False, max_sequences_per_csv=MAX_SEQ_PER_CSV
    )

    print(f"Total sequences: {len(train_dataset)} (train), {len(val_dataset)} (val)")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nERROR: Empty dataset! Check that the CSVs contain valid data.")
        exit(1)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, persistent_workers=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, persistent_workers=True, pin_memory=True
    )

    # Model
    model = DualStreamBiLSTMRegressor().to(DEVICE)

    print(f"\n=== Model ===")
    print(f"Task: Regression (convolutional layer count)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & loss
    criterion = nn.MSELoss()  # mean squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_r2s, val_r2s = [], []

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")

        # --- Training ---
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for x_tex, x_async, yb in tqdm(train_loader, desc="  Training", unit="batch"):
            x_tex = x_tex.to(DEVICE)
            x_async = x_async.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(x_tex, x_async)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_tex.size(0)
            all_preds.append(outputs.detach())
            all_labels.append(yb)

        train_loss = total_loss / len(train_dataset)

        # Compute training metrics
        train_preds = torch.cat(all_preds)
        train_targets = torch.cat(all_labels)
        train_metrics = calculate_regression_metrics(train_preds, train_targets)

        # --- Validation ---
        model.eval()
        val_loss_total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x_tex, x_async, yb in tqdm(val_loader, desc="  Validation", unit="batch"):
                x_tex = x_tex.to(DEVICE)
                x_async = x_async.to(DEVICE)
                yb = yb.to(DEVICE)

                outputs = model(x_tex, x_async)
                loss = criterion(outputs, yb)

                val_loss_total += loss.item() * x_tex.size(0)
                val_preds.append(outputs)
                val_labels.append(yb)

        val_loss = val_loss_total / len(val_dataset)

        # Compute validation metrics
        val_preds_cat = torch.cat(val_preds)
        val_labels_cat = torch.cat(val_labels)
        val_metrics = calculate_regression_metrics(val_preds_cat, val_labels_cat)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_metrics['mae'])
        val_maes.append(val_metrics['mae'])
        train_r2s.append(train_metrics['r2'])
        val_r2s.append(val_metrics['r2'])

        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.2f}, "
              f"R2: {train_metrics['r2']:.4f}, Exact Acc: {train_metrics['exact_accuracy']:.2%}")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.2f}, "
              f"R2: {val_metrics['r2']:.4f}, Exact Acc: {val_metrics['exact_accuracy']:.2%}")

        # Early stopping (based on MAE)
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            epochs_no_improve = 0
            torch.save(model.state_dict(), "layer_count_regressor_best.pt")
            print("  Best model updated!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -----------------------------
    # Final evaluation
    # -----------------------------
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)

    model.load_state_dict(torch.load("layer_count_regressor_best.pt"))
    model.eval()

    final_preds = []
    final_labels = []

    with torch.no_grad():
        for x_tex, x_async, yb in tqdm(val_loader, desc="Final evaluation"):
            x_tex = x_tex.to(DEVICE)
            x_async = x_async.to(DEVICE)
            yb = yb.to(DEVICE)

            outputs = model(x_tex, x_async)
            final_preds.append(outputs)
            final_labels.append(yb)

    final_preds = torch.cat(final_preds)
    final_labels = torch.cat(final_labels)
    final_metrics = calculate_regression_metrics(final_preds, final_labels)

    print(f"\nFinal Metrics:")
    print(f"  MAE (Mean Absolute Error):  {final_metrics['mae']:.2f} layers")
    print(f"  RMSE (Root Mean Squared Error): {final_metrics['rmse']:.2f} layers")
    print(f"  R2 Score: {final_metrics['r2']:.4f}")
    print(f"  Exact Match Accuracy (rounded): {final_metrics['exact_accuracy']:.2%}")
    print(f"  MAPE (Mean Absolute Percentage Error): {final_metrics['mape']:.2f}%")

    # Show a few prediction examples
    print("\n" + "-"*60)
    print("Prediction Examples (first 10):")
    print("-"*60)
    final_preds_np = final_preds.cpu().numpy()
    final_labels_np = final_labels.cpu().numpy()

    for i in range(min(10, len(final_preds_np))):
        pred = final_preds_np[i]
        true = final_labels_np[i]
        error = abs(pred - true)
        print(f"  #{i+1}: Predicted={pred:.1f}, True={true:.0f}, Error={error:.1f}")

    # Per-model analysis
    print("\n" + "-"*60)
    print("Per-Model Analysis:")
    print("-"*60)

    model_predictions = {}
    for model_name, config in MODEL_CONFIG.items():
        num_conv = config['num_conv_layers']
        mask = final_labels_np == num_conv
        if mask.sum() > 0:
            preds_for_model = final_preds_np[mask]
            mae = np.mean(np.abs(preds_for_model - num_conv))
            mean_pred = np.mean(preds_for_model)
            print(f"  {model_name} ({num_conv} layers):")
            print(f"    Mean prediction: {mean_pred:.1f}")
            print(f"    MAE: {mae:.2f}")
            print(f"    Samples: {mask.sum()}")
            model_predictions[model_name] = preds_for_model

    # Save metrics
    with open("layer_count_metrics.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("DUAL-STREAM BiLSTM - CONVOLUTIONAL LAYER COUNT REGRESSION\n")
        f.write("="*60 + "\n\n")
        f.write("Model Configuration:\n")
        for model_name, config in MODEL_CONFIG.items():
            f.write(f"  {model_name}: {config['num_conv_layers']} convolutional layers\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  MAE:  {final_metrics['mae']:.2f} layers\n")
        f.write(f"  RMSE: {final_metrics['rmse']:.2f} layers\n")
        f.write(f"  R2 Score: {final_metrics['r2']:.4f}\n")
        f.write(f"  Exact Match Accuracy: {final_metrics['exact_accuracy']:.2%}\n")
        f.write(f"  MAPE: {final_metrics['mape']:.2f}%\n\n")
        f.write("-"*60 + "\n")
        f.write("Per-Model Analysis:\n")
        f.write("-"*60 + "\n")
        for model_name, config in MODEL_CONFIG.items():
            num_conv = config['num_conv_layers']
            mask = final_labels_np == num_conv
            if mask.sum() > 0:
                preds_for_model = final_preds_np[mask]
                mae = np.mean(np.abs(preds_for_model - num_conv))
                mean_pred = np.mean(preds_for_model)
                f.write(f"\n{model_name} ({num_conv} layers):\n")
                f.write(f"  Mean Prediction: {mean_pred:.2f}\n")
                f.write(f"  MAE: {mae:.2f}\n")
                f.write(f"  Samples: {mask.sum()}\n")

    print("\nMetrics saved to: layer_count_metrics.txt")

    # -----------------------------
    # Visualizations
    # -----------------------------
    fig = plt.figure(figsize=(20, 12))

    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # MAE plot
    plt.subplot(2, 3, 2)
    plt.plot(train_maes, label='Train MAE', linewidth=2)
    plt.plot(val_maes, label='Val MAE', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MAE (layers)", fontsize=12)
    plt.title("Mean Absolute Error", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # R2 score plot
    plt.subplot(2, 3, 3)
    plt.plot(train_r2s, label='Train R2', linewidth=2)
    plt.plot(val_r2s, label='Val R2', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("R2 Score", fontsize=12)
    plt.title("R2 Score Evolution", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Scatter plot: predicted vs actual
    plt.subplot(2, 3, 4)
    plt.scatter(final_labels_np, final_preds_np, alpha=0.5, s=20)

    # Ideal line (y=x)
    min_val = min(final_labels_np.min(), final_preds_np.min())
    max_val = max(final_labels_np.max(), final_preds_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel("True Number of Conv Layers", fontsize=12)
    plt.ylabel("Predicted Number of Conv Layers", fontsize=12)
    plt.title("Predicted vs Actual", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Error distribution
    plt.subplot(2, 3, 5)
    errors = final_preds_np - final_labels_np
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.xlabel("Prediction Error (layers)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Error Distribution", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Box plot per model
    plt.subplot(2, 3, 6)
    if model_predictions:
        box_data = []
        box_labels = []
        for model_name in sorted(MODEL_CONFIG.keys(), key=lambda x: MODEL_CONFIG[x]['num_conv_layers']):
            if model_name in model_predictions:
                box_data.append(model_predictions[model_name])
                box_labels.append(f"{model_name}\n({MODEL_CONFIG[model_name]['num_conv_layers']})")

        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        plt.ylabel("Predicted Conv Layers", fontsize=12)
        plt.title("Predictions Distribution by Model", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("layer_count_regression_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining complete!")
    print(f"Best Validation MAE: {best_val_mae:.2f} layers")
    print(f"Final Validation MAE: {final_metrics['mae']:.2f} layers")
    print(f"Final R2 Score: {final_metrics['r2']:.4f}")
    print(f"\nFiles saved:")
    print(f"  - layer_count_regressor_best.pt (model)")
    print(f"  - layer_count_config.txt (model configuration)")
    print(f"  - layer_count_metrics.txt (detailed metrics)")
    print(f"  - layer_count_regression_metrics.png (plots)")
    print(f"  - csv_pairs_inference_layer_count.txt (inference pairs)")
