import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import requests

# -----------------------------
# 0. Dataset selection
# -----------------------------
DATASET_NAME = "tiny_shakespeare"  # "mnist_seq" | "tiny_shakespeare"

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# 2. Dataset definitions
# =====================================================

# ---------- MNIST as sequence ----------
def get_mnist_seq_loaders(batch_size=128):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./datasets", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./datasets", train=False, download=True, transform=transform)

    def collate(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x.squeeze(0))  # (28,28)
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, test_loader, 28, 10, "classification"


# ---------- Tiny Shakespeare ----------
class CharDataset(Dataset):
    def __init__(self, text, seq_len=10):
        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        self.data = [self.char2idx[c] for c in text]
        self.inputs = []
        self.targets = []
        for i in range(len(self.data) - seq_len):
            self.inputs.append(torch.tensor(self.data[i:i + seq_len]))
            self.targets.append(torch.tensor(self.data[i + seq_len]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_tiny_shakespeare_loaders(batch_size=128, seq_len=10):
    # download the text if not already cached
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    txt = requests.get(url).text

    dataset = CharDataset(txt, seq_len=seq_len)
    split = int(len(dataset) * 0.9)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.vocab_size, dataset.vocab_size, "classification"  # character-level classification


# =====================================================
# 3. Dataset switch
# =====================================================
if DATASET_NAME == "mnist_seq":
    train_loader, test_loader, INPUT_SIZE, NUM_CLASSES, TASK = get_mnist_seq_loaders()
    embedding = False
elif DATASET_NAME == "tiny_shakespeare":
    train_loader, test_loader, INPUT_SIZE, NUM_CLASSES, TASK = get_tiny_shakespeare_loaders()
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
    hidden_size=256,
    output_size=NUM_CLASSES,
    embedding=embedding
).to(device)

# =====================================================
# 5. Loss + optimizer
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =====================================================
# 6. Training loop
# =====================================================
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    start = time.time()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        _, preds = out.max(1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()
        acc = correct / total
        loop.set_postfix(loss=loss_sum / total, acc=acc)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = out.max(1)
            val_total += y.size(0)
            val_correct += preds.eq(y).sum().item()
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f} | Time: {time.time()-start:.1f}s")
