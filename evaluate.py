"""
Quantitative evaluation of generated MIDI files.
Metrics:
  1. Pitch class histogram per emotion (shows harmonic differences)
  2. Note density (notes per second)
  3. Mean pitch and pitch range per emotion
  4. Perplexity per emotion on test set
"""
import os
import numpy as np
import pretty_midi
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from model import EmoMusicTransformer
from dataset import EMOPIADataset
from torch.utils.data import DataLoader

MIDI_DIR  = './outputs/midi'
PLOTS_DIR = './outputs/plots'
PKL_PATH  = './data/sequences_v2.pkl'
CKPT_PATH = './checkpoints/model_best.pt'
os.makedirs(PLOTS_DIR, exist_ok=True)

EMOTION_NAMES = ['Happy', 'Relaxed', 'Sad', 'Tense']
COLORS        = ['#FFD700', '#90EE90', '#6495ED', '#FF6B6B']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = EmoMusicTransformer().to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# ---------------------------------------------------------------
# Metric 1: Pitch class histogram, note density, mean pitch
# ---------------------------------------------------------------
print("--- Metric 1: Musical Feature Analysis ---")

pitch_histograms = {}
note_densities   = {}
mean_pitches     = {}
pitch_ranges     = {}

for emo_name in EMOTION_NAMES:
    emo_dir    = Path(MIDI_DIR) / emo_name
    midi_files = list(emo_dir.glob('*.mid'))

    all_pitches  = []
    all_densities= []

    for mf in midi_files:
        mid   = pretty_midi.PrettyMIDI(str(mf))
        notes = mid.instruments[0].notes if mid.instruments else []
        if not notes:
            continue
        duration = mid.get_end_time()
        pitches  = [n.pitch for n in notes]
        all_pitches.extend(pitches)
        all_densities.append(len(notes) / max(duration, 0.1))

    # Pitch class histogram (12 pitch classes)
    pitch_classes = [p % 12 for p in all_pitches]
    hist = np.zeros(12)
    for pc in pitch_classes:
        hist[pc] += 1
    hist = hist / hist.sum() if hist.sum() > 0 else hist

    pitch_histograms[emo_name] = hist
    note_densities[emo_name]   = np.mean(all_densities)
    mean_pitches[emo_name]     = np.mean(all_pitches)
    pitch_ranges[emo_name]     = np.max(all_pitches) - np.min(all_pitches)

    print(f"\n{emo_name}:")
    print(f"  Note density:  {note_densities[emo_name]:.2f} notes/sec")
    print(f"  Mean pitch:    {mean_pitches[emo_name]:.1f} (MIDI)")
    print(f"  Pitch range:   {pitch_ranges[emo_name]} semitones")

# Plot pitch class histograms
pitch_class_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (emo_name, color) in enumerate(zip(EMOTION_NAMES, COLORS)):
    axes[i].bar(pitch_class_names, pitch_histograms[emo_name], color=color, alpha=0.8)
    axes[i].set_title(f'{emo_name}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Pitch Class')
    axes[i].set_ylabel('Frequency')
    axes[i].set_ylim(0, 0.2)
plt.suptitle('Pitch Class Distribution per Emotion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/pitch_histograms.png', dpi=150)
print(f"\nPitch histogram saved.")

# Plot note density comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].bar(EMOTION_NAMES, [note_densities[e] for e in EMOTION_NAMES], color=COLORS)
axes[0].set_title('Note Density (notes/sec)', fontweight='bold')
axes[0].set_ylabel('Notes per second')

axes[1].bar(EMOTION_NAMES, [mean_pitches[e] for e in EMOTION_NAMES], color=COLORS)
axes[1].set_title('Mean Pitch (MIDI)', fontweight='bold')
axes[1].set_ylabel('MIDI pitch value')

axes[2].bar(EMOTION_NAMES, [pitch_ranges[e] for e in EMOTION_NAMES], color=COLORS)
axes[2].set_title('Pitch Range (semitones)', fontweight='bold')
axes[2].set_ylabel('Semitones')

plt.suptitle('Musical Feature Comparison Across Emotions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/musical_features.png', dpi=150)
print("Musical features plot saved.")

# ---------------------------------------------------------------
# Metric 2: Perplexity per emotion on test set
# ---------------------------------------------------------------
print("\n--- Metric 2: Per-Emotion Perplexity on Test Set ---")

criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
test_ds   = EMOPIADataset(PKL_PATH, split='test')

emo_losses  = {i: [] for i in range(4)}
emo_tokens  = {i: 0 for i in range(4)}

with torch.no_grad():
    for x, y, label in torch.utils.data.DataLoader(test_ds, batch_size=16):
        x, y, label = x.to(device), y.to(device), label.to(device)
        logits = model(x, label)
        B, T, V = logits.shape
        for i in range(4):
            mask = label == i
            if mask.sum() == 0:
                continue
            loss = criterion(
                logits[mask].reshape(-1, V),
                y[mask].reshape(-1)
            ).item()
            non_pad = (y[mask] != 0).sum().item()
            emo_losses[i].append(loss)
            emo_tokens[i] += non_pad

print("\nPer-emotion perplexity:")
emo_ppls = {}
for i, emo_name in enumerate(EMOTION_NAMES):
    if emo_tokens[i] > 0:
        avg_loss = sum(emo_losses[i]) / emo_tokens[i]
        ppl = np.exp(avg_loss)
        emo_ppls[emo_name] = ppl
        print(f"  {emo_name}: ppl={ppl:.2f}")

# Plot per-emotion perplexity
plt.figure(figsize=(8, 5))
plt.bar(list(emo_ppls.keys()), list(emo_ppls.values()), color=COLORS)
plt.title('Per-Emotion Perplexity on Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Perplexity (lower = better)')
plt.axhline(y=np.mean(list(emo_ppls.values())), color='black',
            linestyle='--', label=f'Mean: {np.mean(list(emo_ppls.values())):.2f}')
plt.legend()
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/per_emotion_perplexity.png', dpi=150)
print("Per-emotion perplexity plot saved.")

print("\n========== EVALUATION SUMMARY ==========")
print(f"{'Emotion':<12} {'Density':>10} {'Mean Pitch':>12} {'Range':>8} {'PPL':>8}")
print("-" * 55)
for emo in EMOTION_NAMES:
    print(f"{emo:<12} {note_densities[emo]:>10.2f} {mean_pitches[emo]:>12.1f} "
          f"{pitch_ranges[emo]:>8} {emo_ppls.get(emo, 0):>8.2f}")
print("=========================================")
