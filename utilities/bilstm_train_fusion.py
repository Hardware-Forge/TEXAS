import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import random

# -----------------------------
# Parameters
# -----------------------------
DATA_DIRS = ["src/Attack/TEX/final_plots3050", "src/Attack/Async/final_plots3050"]
SEQ_LEN = 256
STRIDE = 16
BATCH_SIZE = 256
NUM_EPOCHS = 150
LR = 1e-3
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_PER_CSV = 300000  # 100000 is fine too
IGNORE_CSVS = []

# -----------------------------
# Class mapping
# -----------------------------
CLASS_MAP = {
    0: "CNN",
    2: "Transformer",
    3: "Diffusion",
    4: "RNN",
    5: "Mining"
}

CLASS_REMAP = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}

# -----------------------------
# Dual-Stream Dataset
# -----------------------------
class DualStreamLatencyDataset(Dataset):
    """
    Loads pairs of sequences (TEX, Async) for the same class.
    Each sample contains one TEX sequence and one Async sequence.
    """
    def __init__(self, csv_pairs, seq_len=256, stride=16, train=True,
                 split_ratio=0.8, max_sequences_per_csv=50000):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []  # list of tuples: (tex_block, tex_start, async_block, async_start)
        self.labels = []

        for tex_csv, async_csv in tqdm(csv_pairs, desc="Loading CSV pairs", unit="pair"):
            # Load TEX
            df_tex = pd.read_csv(tex_csv)
            latencies_tex = df_tex['latency_cycles'].values
            raw_label = int(df_tex['class'].iloc[0])
            label = CLASS_REMAP[raw_label]

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
                self.labels.append(label)

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
                self.labels[idx])

# -----------------------------
# Dual-Stream BiLSTM Model
# -----------------------------
class DualStreamBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()

        # TEX stream
        self.lstm_tex = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True)

        # Async stream
        self.lstm_async = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True)

        # Fusion layer
        self.fc_fusion = nn.Linear(hidden_size * 4, 128)  # 4 = 2 streams * 2 directions
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x_tex, x_async):
        # Add the input_size dimension
        x_tex = x_tex.unsqueeze(-1)      # (batch, seq_len, 1)
        x_async = x_async.unsqueeze(-1)  # (batch, seq_len, 1)

        # Process the TEX stream
        out_tex, _ = self.lstm_tex(x_tex)
        feat_tex = out_tex[:, -1, :]  # last timestep

        # Process the Async stream
        out_async, _ = self.lstm_async(x_async)
        feat_async = out_async[:, -1, :]  # last timestep

        # Concatenate features
        combined = torch.cat([feat_tex, feat_async], dim=-1)

        # Classification head
        x = torch.relu(self.fc_fusion(combined))
        x = self.bn(x)
        x = self.dropout(x)
        out = self.fc_out(x)

        return out

# -----------------------------
# Helper to build CSV pairs
# -----------------------------
def create_csv_pairs(data_dirs, ignore_csvs):
    """
    Builds (TEX, Async) CSV pairs for the same class.
    Pairs files by name: async_results_* with tex_results_*.
    Returns: (train_val_pairs, infer_pairs)
    """
    SEED = 42
    random.seed(SEED)

    # Dict: class_name -> {tex: {key: filepath}, async: {key: filepath}}
    class_csvs = {}

    for data_dir in data_dirs:
        class_dirs = glob.glob(os.path.join(data_dir, "**", "plots", "*"), recursive=True)
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        for class_dir in class_dirs:
            csvs = glob.glob(os.path.join(class_dir, "*.csv"))
            if not csvs:
                continue

            folder_name = os.path.basename(class_dir).lower()

            if folder_name in ignore_csvs:
                continue

            # Initialize the dict for this class
            if folder_name not in class_csvs:
                class_csvs[folder_name] = {'tex': {}, 'async': {}}

            # Determine whether this is TEX or Async
            stream = 'tex' if 'TEX' in data_dir else 'async'

            # Build a dict with keys for pairing
            for csv_file in csvs:
                basename = os.path.basename(csv_file)

                # Extract the key from the file name
                if basename.startswith('tex_results_'):
                    key = basename.replace('tex_results_', '')
                    class_csvs[folder_name]['tex'][key] = csv_file
                elif basename.startswith('async_results_'):
                    key = basename.replace('async_results_', '')
                    class_csvs[folder_name]['async'][key] = csv_file

    # Build the pairs
    train_val_pairs = []
    infer_pairs = []

    print("\n" + "="*60)
    print("BUILDING CSV PAIRS (MATCHED BY FILE NAME)")
    print("="*60)

    for class_name, streams in class_csvs.items():
        tex_dict = streams['tex']
        async_dict = streams['async']

        print(f"\nClass {class_name}:")
        print(f"  TEX files found: {len(tex_dict)}")
        print(f"  Async files found: {len(async_dict)}")

        if not tex_dict or not async_dict:
            print(f"  Skipped (missing files in one of the two streams)")
            continue

        # Find the common keys (files present in both TEX and Async)
        common_keys = set(tex_dict.keys()) & set(async_dict.keys())

        print(f"  Valid pairs found: {len(common_keys)}")

        if not common_keys:
            print(f"  No matching pairs found!")
            print(f"  TEX key examples: {list(tex_dict.keys())[:3]}")
            print(f"  Async key examples: {list(async_dict.keys())[:3]}")
            continue

        # Split train/test based on the key name
        train_keys = [k for k in common_keys if "train" in k.lower()]
        test_keys = [k for k in common_keys if "test" in k.lower()]

        print(f"  Train keys: {len(train_keys)}")
        print(f"  Test keys: {len(test_keys)}")

        # Pick aside files for inference (one train and one test, if available)
        infer_train_key = random.choice(train_keys) if train_keys else None
        infer_test_key = random.choice(test_keys) if test_keys else None

        # Add the inference pairs
        if infer_train_key:
            infer_pairs.append((
                tex_dict[infer_train_key],
                async_dict[infer_train_key]
            ))
            print(f"  Inference train pair: {infer_train_key}")

        if infer_test_key:
            infer_pairs.append((
                tex_dict[infer_test_key],
                async_dict[infer_test_key]
            ))
            print(f"  Inference test pair: {infer_test_key}")

        # Build the train/val pairs from the remaining keys
        remaining_keys = common_keys - {infer_train_key, infer_test_key}
        remaining_keys = sorted(remaining_keys)  # for consistency

        for key in remaining_keys:
            train_val_pairs.append((
                tex_dict[key],
                async_dict[key]
            ))

        print(f"  -> {len(remaining_keys)} pairs for train/val")

        # Show a few pairing examples
        if remaining_keys:
            print(f"  Train/val pairing examples:")
            for i, key in enumerate(list(remaining_keys)[:3]):
                print(f"    {i+1}. {key}")
                print(f"       TEX:   {os.path.basename(tex_dict[key])}")
                print(f"       Async: {os.path.basename(async_dict[key])}")

    print(f"\nTotal train/val pairs: {len(train_val_pairs)}")
    print(f"Total inference pairs: {len(infer_pairs)}")

    # Final shuffle to mix classes together
    random.shuffle(train_val_pairs)

    return train_val_pairs, infer_pairs

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()

    # Build CSV pairs
    train_val_pairs, infer_pairs = create_csv_pairs(DATA_DIRS, IGNORE_CSVS)

    if not train_val_pairs:
        print("\nERROR: No training pairs found! Check:")
        print("  1. The paths in DATA_DIRS")
        print("  2. The folders in IGNORE_CSVS")
        print("  3. That .csv files exist in those folders")
        print("  4. That the file names follow the pattern:")
        print("     - tex_results_*.csv")
        print("     - async_results_*.csv")
        exit(1)

    print(f"\n=== Summary ===")
    print(f"Train/val pairs: {len(train_val_pairs)}")
    print(f"Inference pairs: {len(infer_pairs)}")

    # Save the inference pairs
    with open("csv_pairs_not_used_for_training.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("CSV PAIRS NOT USED FOR TRAINING (INFERENCE SET)\n")
        f.write("="*60 + "\n\n")
        for tex_csv, async_csv in infer_pairs:
            f.write(f"TEX:   {os.path.basename(tex_csv)}\n")
            f.write(f"Async: {os.path.basename(async_csv)}\n")
            f.write(f"Full paths:\n")
            f.write(f"  TEX:   {tex_csv}\n")
            f.write(f"  Async: {async_csv}\n\n")

    # Datasets
    train_dataset = DualStreamLatencyDataset(
        train_val_pairs, seq_len=SEQ_LEN, stride=STRIDE,
        train=True, max_sequences_per_csv=MAX_SEQ_PER_CSV
    )
    val_dataset = DualStreamLatencyDataset(
        train_val_pairs, seq_len=SEQ_LEN, stride=STRIDE,
        train=False, max_sequences_per_csv=MAX_SEQ_PER_CSV
    )

    print(f"\nTotal sequences: {len(train_dataset)} (train), {len(val_dataset)} (val)")

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
    NUM_CLASSES = len(CLASS_MAP)
    model = DualStreamBiLSTM(num_classes=NUM_CLASSES).to(DEVICE)

    print(f"\n=== Model ===")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = 0
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    train_accs, val_accs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

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
            logits = model(x_tex, x_async)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_tex.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        train_loss = total_loss / len(train_dataset)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_acc = accuracy_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        # --- Validation ---
        model.eval()
        val_loss_total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x_tex, x_async, yb in tqdm(val_loader, desc="  Validation", unit="batch"):
                x_tex = x_tex.to(DEVICE)
                x_async = x_async.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(x_tex, x_async)
                loss = criterion(logits, yb)

                val_loss_total += loss.item() * x_tex.size(0)
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_loss = val_loss_total / len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "dual_stream_bilstm_best.pt")
            print("  Best model updated!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -----------------------------
    # Final evaluation on validation set
    # -----------------------------
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)

    model.load_state_dict(torch.load("dual_stream_bilstm_best.pt"))
    model.eval()

    final_preds = []
    final_labels = []

    with torch.no_grad():
        for x_tex, x_async, yb in tqdm(val_loader, desc="Final evaluation"):
            x_tex = x_tex.to(DEVICE)
            x_async = x_async.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(x_tex, x_async)
            preds = logits.argmax(dim=-1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(yb.cpu().numpy())

    # Calculate final metrics
    final_acc = accuracy_score(final_labels, final_preds)
    final_f1 = f1_score(final_labels, final_preds, average='weighted')
    final_precision = precision_score(final_labels, final_preds, average='weighted', zero_division=0)
    final_recall = recall_score(final_labels, final_preds, average='weighted', zero_division=0)

    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  F1 Score:  {final_f1:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")

    # Per-class metrics
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    class_names = [CLASS_MAP[k] for k in sorted(CLASS_MAP.keys())]
    print(classification_report(final_labels, final_preds,
                                target_names=class_names,
                                digits=4))

    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)

    # Save metrics to file
    with open("dual_stream_metrics.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("DUAL-STREAM BiLSTM - FINAL METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {final_acc:.4f} ({final_acc*100:.2f}%)\n")
        f.write(f"  F1 Score:  {final_f1:.4f}\n")
        f.write(f"  Precision: {final_precision:.4f}\n")
        f.write(f"  Recall:    {final_recall:.4f}\n\n")
        f.write("-"*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(classification_report(final_labels, final_preds,
                                     target_names=class_names,
                                     digits=4))
        f.write("\n" + "-"*60 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-"*60 + "\n")
        f.write(str(cm))

    print("\nMetrics saved to: dual_stream_metrics.txt")

    # Plot results
    fig = plt.figure(figsize=(18, 10))

    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(val_accs, label='Val Accuracy', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # F1 score plot
    plt.subplot(2, 3, 3)
    plt.plot(train_f1s, label='Train F1', linewidth=2)
    plt.plot(val_f1s, label='Val F1', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Training and Validation F1 Score", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Precision plot
    plt.subplot(2, 3, 4)
    plt.plot(train_precisions, label='Train Precision', linewidth=2)
    plt.plot(val_precisions, label='Val Precision', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Training and Validation Precision", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Recall plot
    plt.subplot(2, 3, 5)
    plt.plot(train_recalls, label='Train Recall', linewidth=2)
    plt.plot(val_recalls, label='Val Recall', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Training and Validation Recall", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Confusion matrix
    plt.subplot(2, 3, 6)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig("dual_stream_training_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Separate confusion matrix plot (larger)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=14, fontweight='bold')
    plt.title("Confusion Matrix - Dual-Stream BiLSTM", fontsize=16, fontweight='bold')

    # Add per-class accuracy on the side
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracies):
        plt.text(len(class_names) + 0.5, i + 0.5, f'{acc:.2%}',
                ha='left', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("confusion_matrix_detailed.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining complete!")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"\nFiles saved:")
    print(f"  - dual_stream_bilstm_best.pt (model)")
    print(f"  - csv_pairs_not_used_for_training.txt (inference pairs)")
    print(f"  - dual_stream_metrics.txt (detailed metrics)")
    print(f"  - dual_stream_training_metrics.png (training plots)")
    print(f"  - confusion_matrix_detailed.png (confusion matrix)")
