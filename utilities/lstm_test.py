import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import requests

# -----------------------------
# 0. Dataset selection
# -----------------------------
DATASET_NAME = "mnist_seq"   # "mnist_seq" | "tiny_shakespeare"

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 2. Dataset definitions
# =====================================================

# ---------- MNIST as sequence ----------
def get_mnist_seq_loader(batch_size=128):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST("./datasets", train=False, download=True, transform=transform)

    def collate(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x.squeeze(0))  # (28,28)
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return test_loader, 28, 10, "classification"


# ---------- Tiny Shakespeare ----------
class CharDataset(Dataset):
    def __init__(self, text, seq_len=10):
        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        data = [self.char2idx[c] for c in text]
        self.inputs = []
        self.targets = []
        for i in range(len(data) - seq_len):
            self.inputs.append(torch.tensor(data[i:i + seq_len]))
            self.targets.append(torch.tensor(data[i + seq_len]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_shakespeare_loader(batch_size=64, seq_len=10):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    txt = requests.get(url).text

    dataset = CharDataset(txt, seq_len=seq_len)
    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    _, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return test_loader, dataset.vocab_size, dataset.vocab_size, "classification"


# -----------------------------
# 3. Dataset switch
# -----------------------------
if DATASET_NAME == "mnist_seq":
    test_loader, INPUT_SIZE, NUM_CLASSES, TASK = get_mnist_seq_loader()
    embedding = False
elif DATASET_NAME == "tiny_shakespeare":
    test_loader, INPUT_SIZE, NUM_CLASSES, TASK = get_shakespeare_loader()
    embedding = True

# =====================================================
# 4. Model (LSTM)
# =====================================================
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding=embedding):
        super().__init__()
        self.embedding = embedding
        if embedding:
            self.embed = nn.Embedding(input_size, hidden_size)
            input_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.embedding:
            x = self.embed(x)
        _, (h, _) = self.rnn(x)
        return self.fc(h[-1])


model = RNNModel(
    input_size=INPUT_SIZE,
    hidden_size=128,
    output_size=NUM_CLASSES,
    embedding=(DATASET_NAME == "tiny_shakespeare")
).to(device)

# Load trained weights here if you have them, e.g.:
# model.load_state_dict(torch.load("rnn_trained_shakespeare.pth"))

model.eval()

# =====================================================
# 5. Timing loop
# =====================================================
DURATION_SECONDS = 180
start_time = time.time()
iters = 0
total, correct = 0, 0

print(f"Starting inference for {DURATION_SECONDS}s on {DATASET_NAME}...")

with torch.no_grad():
    while time.time() - start_time < DURATION_SECONDS:
        loop = tqdm(test_loader, desc=f"Iteration {iters+1}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if TASK == "classification":
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                loop.set_postfix(acc=f"{correct/total:.4f}", iters=iters)
            else:  # regression
                total += y.size(0)
                loop.set_postfix(samples=total, iters=iters)

            iters += 1
            if device.type == "cuda":
                torch.cuda.synchronize()

            if time.time() - start_time >= DURATION_SECONDS:
                break

elapsed = time.time() - start_time
print("\nInference finished")
print(f"Iterations completed: {iters}")
print(f"Processed samples: {total}")
if TASK == "classification":
    print(f"Average accuracy: {correct/total:.4f}")
print(f"Total time elapsed: {elapsed:.1f}s")
