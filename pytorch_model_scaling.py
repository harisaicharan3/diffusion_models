import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# ---- Dummy Dataset: character-level toy corpus ----
class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = [self.stoi[c] for c in text]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ---- Mini Transformer Model ----
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, block_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

# ---- Training Loop ----
def train_model(model, dataloader, optimizer, device, epochs=3):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
    return np.mean(losses[-1:])  # return final average loss

# ---- Scaling Experiment ----
def run_scaling_experiment():
    text = "hello world " * 1000
    block_size = 16
    batch_size = 32
    dataset = CharDataset(text, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    configs = [
        {"d_model": 64, "n_layers": 2},
        {"d_model": 128, "n_layers": 2},
        {"d_model": 128, "n_layers": 4},
        {"d_model": 256, "n_layers": 4},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for cfg in configs:
        model = MiniTransformer(
            vocab_size=dataset.vocab_size,
            d_model=cfg["d_model"],
            n_heads=4,
            n_layers=cfg["n_layers"],
            block_size=block_size,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        param_count = sum(p.numel() for p in model.parameters())
        loss = train_model(model, dataloader, optimizer, device)
        print(f"Params: {param_count/1e3:.1f}K | Loss: {loss:.4f}")
        results.append((param_count, loss))

    # Plot
    params, losses = zip(*results)
    plt.figure()
    plt.plot(np.log10(params), losses, marker='o')
    plt.xlabel("log10(# Parameters)")
    plt.ylabel("Loss")
    plt.title("Scaling Law: Loss vs Model Size")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_scaling_experiment()
