"""
Generate MIDI from trained EmoMusicTransformer.
Supports:
  1. Single emotion generation
  2. Emotion trajectory generation (novel contribution)
Converts token sequences back to MIDI using pretty_midi.
"""
import os
import torch
import numpy as np
import pretty_midi
from model import EmoMusicTransformer

# Token config (must match preprocess.py)
PAD, BOS, EOS   = 0, 1, 2
PITCH_OFFSET     = 7
VELOCITY_OFFSET  = 135
DURATION_OFFSET  = 167
TIMESHIFT_OFFSET = 199
N_VELOCITY       = 32
N_DURATION       = 32
N_TIMESHIFT      = 64

EMOTION_NAMES = {0:'Happy', 1:'Relaxed', 2:'Sad', 3:'Tense'}
CKPT_PATH     = './checkpoints/model_best.pt'
OUTPUT_DIR    = './outputs/midi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = EmoMusicTransformer().to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()
print(f"Loaded model from {CKPT_PATH}")

def unbin(bin_idx, min_val, max_val, n_bins):
    return min_val + (bin_idx / (n_bins - 1)) * (max_val - min_val)

def tokens_to_midi(tokens, tempo=120):
    """Convert flat token sequence back to pretty_midi.PrettyMIDI."""
    mid  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)  # Grand Piano

    i = 0
    current_time = 0.0
    pending_timeshift = 0.0
    pending_pitch     = None
    pending_duration  = None
    pending_velocity  = None

    while i < len(tokens):
        tok = tokens[i]
        if tok in (PAD, BOS, EOS):
            i += 1
            continue

        if TIMESHIFT_OFFSET <= tok < TIMESHIFT_OFFSET + N_TIMESHIFT:
            ts_bin = tok - TIMESHIFT_OFFSET
            pending_timeshift = unbin(ts_bin, 0.0, 1.0, N_TIMESHIFT)

        elif PITCH_OFFSET <= tok < PITCH_OFFSET + 128:
            pending_pitch = tok - PITCH_OFFSET

        elif DURATION_OFFSET <= tok < DURATION_OFFSET + N_DURATION:
            dur_bin = tok - DURATION_OFFSET
            pending_duration = unbin(dur_bin, 0.01, 4.0, N_DURATION)

        elif VELOCITY_OFFSET <= tok < VELOCITY_OFFSET + N_VELOCITY:
            vel_bin = tok - VELOCITY_OFFSET
            pending_velocity = int(unbin(vel_bin, 0, 127, N_VELOCITY))

            # We have all 4 components — emit note
            if pending_pitch is not None and pending_duration is not None:
                current_time += pending_timeshift
                note = pretty_midi.Note(
                    velocity=max(1, min(127, pending_velocity)),
                    pitch=max(0, min(127, pending_pitch)),
                    start=current_time,
                    end=current_time + pending_duration
                )
                inst.notes.append(note)
                pending_pitch = pending_duration = pending_timeshift = 0.0
        i += 1

    mid.instruments.append(inst)
    return mid

# ---------------------------------------------------------------
# 1. Single emotion generation — 4 samples per emotion
# ---------------------------------------------------------------
print("\n--- Single Emotion Generation ---")
for q in range(4):
    emo_name = EMOTION_NAMES[q]
    emo_dir  = os.path.join(OUTPUT_DIR, emo_name)
    os.makedirs(emo_dir, exist_ok=True)

    for i in range(4):
        print(f"  Generating {emo_name} sample {i+1}/4...")
        tokens = model.generate(q, device, max_new_tokens=512, top_p=0.9)
        mid    = tokens_to_midi(tokens)
        path   = os.path.join(emo_dir, f"{emo_name}_{i+1}.mid")
        mid.write(path)
        print(f"    Saved: {path} ({len(mid.instruments[0].notes)} notes)")

# ---------------------------------------------------------------
# 2. Emotion trajectory generation (novel contribution)
# ---------------------------------------------------------------
print("\n--- Emotion Trajectory Generation ---")
trajectories = [
    ([0, 1, 2],    "Happy_to_Relaxed_to_Sad"),
    ([3, 2, 1],    "Tense_to_Sad_to_Relaxed"),
    ([0, 3, 2, 1], "Happy_to_Tense_to_Sad_to_Relaxed"),
    ([2, 0],       "Sad_to_Happy"),
    ([1, 3],       "Relaxed_to_Tense"),
]

traj_dir = os.path.join(OUTPUT_DIR, 'trajectories')
os.makedirs(traj_dir, exist_ok=True)

for traj, name in trajectories:
    print(f"  Generating trajectory: {name}...")
    tokens = model.generate_trajectory(traj, device,
                                        tokens_per_segment=128, top_p=0.9)
    mid  = tokens_to_midi(tokens)
    path = os.path.join(traj_dir, f"{name}.mid")
    mid.write(path)
    traj_str = ' -> '.join([EMOTION_NAMES[e] for e in traj])
    print(f"    Saved: {path} | trajectory: {traj_str} | notes: {len(mid.instruments[0].notes)}")

print("\nGeneration complete.")
