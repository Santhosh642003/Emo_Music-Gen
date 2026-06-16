"""
Tokenize EMOPIA MIDI files into note-level token sequences.
Improvements:
  - Time shift capped at 1.0s (was 4.0s) — fixes silence issue
  - Data augmentation: transpose each file to all 12 keys
  - Same token format: TIMESHIFT, PITCH, DURATION, VELOCITY
"""
import os, pickle
import numpy as np
import pretty_midi
from pathlib import Path
from copy import deepcopy

MIDI_DIR   = './data/EMOPIA_1.0/midis'
OUT_PATH   = './data/sequences_v2.pkl'
MAX_SEQ    = 512
N_VELOCITY = 32
N_DURATION = 32
N_TIMESHIFT= 64

PAD, BOS, EOS    = 0, 1, 2
PITCH_OFFSET     = 7
VELOCITY_OFFSET  = 135
DURATION_OFFSET  = 167
TIMESHIFT_OFFSET = 199
VOCAB_SIZE       = 263

def bin_value(val, min_val, max_val, n_bins):
    val = max(min_val, min(max_val, val))
    return int((val - min_val) / (max_val - min_val) * (n_bins - 1))

def midi_to_tokens(midi_path, transpose=0):
    mid = pretty_midi.PrettyMIDI(str(midi_path))
    if not mid.instruments:
        return None
    notes = []
    for inst in mid.instruments:
        if not inst.is_drum:
            for n in inst.notes:
                pitch = n.pitch + transpose
                if 0 <= pitch <= 127:
                    notes.append((n.start, pitch, n.end - n.start, n.velocity))
    if not notes:
        return None
    notes.sort(key=lambda x: x[0])

    tokens = [BOS]
    prev_start = 0.0
    for start, pitch, duration, velocity in notes[:MAX_SEQ]:
        # Time shift capped at 1.0s
        ts = max(0.0, min(start - prev_start, 1.0))
        ts_bin  = bin_value(ts, 0.0, 1.0, N_TIMESHIFT)
        dur_bin = bin_value(min(duration, 4.0), 0.01, 4.0, N_DURATION)
        vel_bin = bin_value(velocity, 0, 127, N_VELOCITY)

        tokens.append(TIMESHIFT_OFFSET + ts_bin)
        tokens.append(PITCH_OFFSET + pitch)
        tokens.append(DURATION_OFFSET + dur_bin)
        tokens.append(VELOCITY_OFFSET + vel_bin)
        prev_start = start

    tokens.append(EOS)
    return tokens

def get_quadrant(filename):
    return int(Path(filename).stem[1]) - 1

# Process all files with 12 transpositions
sequences = []
midi_files = sorted(Path(MIDI_DIR).glob('*.mid'))
print(f"Processing {len(midi_files)} MIDI files x 12 transpositions...")

for i, midi_path in enumerate(midi_files):
    try:
        quadrant = get_quadrant(midi_path.name)
        for transpose in range(-6, 6):  # -6 to +5 semitones = 12 keys
            tokens = midi_to_tokens(midi_path, transpose=transpose)
            if tokens and len(tokens) > 10:
                sequences.append((tokens, quadrant))
    except Exception as e:
        pass
    if (i+1) % 100 == 0:
        print(f"  [{i+1}/{len(midi_files)}] sequences={len(sequences)}")

print(f"\nDone: {len(sequences)} sequences")
lengths = [len(s[0]) for s in sequences]
print(f"Length: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.0f}")
by_q = {0:0,1:0,2:0,3:0}
for _,q in sequences: by_q[q]+=1
print("Per quadrant:", by_q)

with open(OUT_PATH, 'wb') as f:
    pickle.dump({'sequences': sequences, 'vocab_size': VOCAB_SIZE}, f)
print(f"Saved to {OUT_PATH}")
