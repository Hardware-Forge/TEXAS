import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
import random

# -----------------------------
# Parameters
# -----------------------------
SEQ_LEN = 256
STRIDE = 16
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_ROWS = 50000

# Model path
MODEL_PATH = "dual_stream_bilstm_best.pt"

# CSV paths - YOU MUST PROVIDE BOTH (TEX and Async)
CSV_TEX = "src/Attack/TEX/plots/MobileNet/tex_results_streaming_MobileNet_train_svhn.csv"
CSV_ASYNC = "src/Attack/Async/plots/MobileNet/async_results_streaming_MobileNet_train_svhn.csv"

# Class mapping (model output -> class name)
REVERSE_CLASS_MAP = {
    0: "CNN",
    1: "Transformer",
    2: "Diffusion",
    3: "RNN",
    4: "Mining"
}

# -----------------------------
# Dual-Stream Dataset for Inference
# -----------------------------
class DualStreamInferenceDataset(Dataset):
    def __init__(self, latencies_tex, latencies_async, seq_len=256, stride=16):
        self.seq_len = seq_len
        self.stride = stride
        self.data = []

        # Use the shorter of the two streams
        min_len = min(len(latencies_tex), len(latencies_async))
        num_seqs = min_len - seq_len

        if num_seqs <= 0:
            raise ValueError("CSV too short for the given sequence length.")

        indices = range(0, num_seqs, stride)
        for i in indices:
            seq_tex = latencies_tex[i:i+seq_len]
            seq_async = latencies_async[i:i+seq_len]
            self.data.append((seq_tex, seq_async))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_tex, seq_async = self.data[idx]

        # Normalize each stream independently
        seq_tex = (seq_tex - seq_tex.mean()) / (seq_tex.std() + 1e-8)
        seq_async = (seq_async - seq_async.mean()) / (seq_async.std() + 1e-8)

        return (torch.tensor(seq_tex, dtype=torch.float32),
                torch.tensor(seq_async, dtype=torch.float32))

# -----------------------------
# Dual-Stream BiLSTM Model
# -----------------------------
class DualStreamBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()

        self.lstm_tex = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True)
        self.lstm_async = nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=True)

        self.fc_fusion = nn.Linear(hidden_size * 4, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, num_classes)

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

        return out

# -----------------------------
# Helper function to read CSV with random block
# -----------------------------
def read_csv_random_block(csv_path, max_rows):
    """Reads a random block of up to max_rows rows from the CSV."""
    total_rows = sum(1 for _ in open(csv_path)) - 1  # subtract header

    if total_rows <= max_rows:
        df = pd.read_csv(csv_path)
    else:
        start_row = random.randint(0, total_rows - max_rows)
        df = pd.read_csv(csv_path, skiprows=range(1, start_row+1), nrows=max_rows)

    return df['latency_cycles'].values

# -----------------------------
# Main Inference
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("DUAL-STREAM BiLSTM INFERENCE")
    print("="*60)

    # Make sure both files exist
    if not os.path.exists(CSV_TEX):
        raise FileNotFoundError(f"TEX CSV not found: {CSV_TEX}")
    if not os.path.exists(CSV_ASYNC):
        raise FileNotFoundError(f"Async CSV not found: {CSV_ASYNC}")

    print(f"\nLoading data...")
    print(f"  TEX:   {CSV_TEX}")
    print(f"  Async: {CSV_ASYNC}")

    # Read both CSVs
    latencies_tex = read_csv_random_block(CSV_TEX, MAX_ROWS)
    latencies_async = read_csv_random_block(CSV_ASYNC, MAX_ROWS)

    print(f"\nData loaded:")
    print(f"  TEX sequences:   {len(latencies_tex)}")
    print(f"  Async sequences: {len(latencies_async)}")

    # Create dataset and dataloader
    dataset = DualStreamInferenceDataset(latencies_tex, latencies_async,
                                         seq_len=SEQ_LEN, stride=STRIDE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"  Total windows:   {len(dataset)}")

    # Load model
    NUM_CLASSES = len(REVERSE_CLASS_MAP)
    model = DualStreamBiLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"\nModel loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"\nRunning inference...")

    # Inference
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x_tex, x_async in tqdm(loader, desc="Inference", unit="batch"):
            x_tex = x_tex.to(DEVICE)
            x_async = x_async.to(DEVICE)

            logits = model(x_tex, x_async)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Majority voting
    counter = Counter(all_preds)
    most_common_class = counter.most_common(1)[0][0]
    class_name = REVERSE_CLASS_MAP[most_common_class]

    # Calculate confidence (average probability for the most common class)
    import numpy as np
    probs_array = np.array(all_probs)
    avg_confidence = probs_array[:, most_common_class].mean()

    # Results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"\nPredicted class: {class_name}")
    print(f"Confidence: {avg_confidence:.2%}")
    print(f"\nVoting distribution:")
    for class_idx, count in counter.most_common():
        percentage = (count / len(all_preds)) * 100
        print(f"  {REVERSE_CLASS_MAP[class_idx]:12s}: {count:5d} votes ({percentage:5.1f}%)")

    # Per-class average probabilities
    print(f"\nAverage probabilities across all windows:")
    avg_probs = probs_array.mean(axis=0)
    for class_idx, prob in enumerate(avg_probs):
        print(f"  {REVERSE_CLASS_MAP[class_idx]:12s}: {prob:.2%}")

    print("\n" + "="*60)
