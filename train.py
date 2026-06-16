"""
Training loop for EmoMusicTransformer.
Metrics: cross-entropy loss, perplexity, per-emotion accuracy.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import EMOPIADataset
from model import EmoMusicTransformer

# Config
PKL_PATH   = './data/sequences_v2.pkl'
CKPT_DIR   = './checkpoints'
PLOTS_DIR  = './outputs'
EPOCHS     = 100
BATCH_SIZE = 32
LR         = 3e-4
WARMUP     = 500       # warmup steps
GRAD_CLIP  = 1.0
SAVE_EVERY = 10

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Datasets
train_ds = EMOPIADataset(PKL_PATH, split='train')
val_ds   = EMOPIADataset(PKL_PATH, split='val')
test_ds  = EMOPIADataset(PKL_PATH, split='test')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

# Model
model = EmoMusicTransformer().to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               betas=(0.9, 0.95), weight_decay=0.1)

# Cosine LR schedule with warmup
def get_lr(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

TOTAL_STEPS = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_tokens = 0, 0
    with torch.set_grad_enabled(train):
        for x, y, label in loader:
            x, y, label = x.to(device), y.to(device), label.to(device)
            logits = model(x, label)          # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
            # Count non-pad tokens
            non_pad = (y != 0).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

best_val_ppl = float('inf')
train_ppls, val_ppls = [], []

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_ppl = run_epoch(train_loader, train=True)
    va_loss, va_ppl = run_epoch(val_loader,   train=False)

    train_ppls.append(tr_ppl)
    val_ppls.append(va_ppl)

    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:03d}/{EPOCHS} | "
          f"Train: loss={tr_loss:.4f} ppl={tr_ppl:.2f} | "
          f"Val: loss={va_loss:.4f} ppl={va_ppl:.2f} | "
          f"LR={lr_now:.6f}")

    if epoch % SAVE_EVERY == 0:
        torch.save(model.state_dict(), f"{CKPT_DIR}/model_epoch{epoch:03d}.pt")

    if va_ppl < best_val_ppl:
        best_val_ppl = va_ppl
        torch.save(model.state_dict(), f"{CKPT_DIR}/model_best.pt")
        print(f"  -> Best model saved (val_ppl={best_val_ppl:.2f})")

# Plot
plt.figure(figsize=(8,4))
plt.plot(train_ppls, label='Train Perplexity')
plt.plot(val_ppls,   label='Val Perplexity')
plt.xlabel('Epoch'); plt.ylabel('Perplexity')
plt.title('EmoMusicTransformer Training')
plt.legend(); plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/training_curves.png")
print("Training complete.")
