"""
PyTorch Dataset for EmoMusicGen.
Returns (input_tokens, target_tokens, emotion_label) for causal LM training.
Supports emotion trajectory: sequence of emotion labels, one per segment.
"""
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_SEQ    = 512
PAD        = 0
VOCAB_SIZE = 263

class EMOPIADataset(Dataset):
    def __init__(self, pkl_path, split='train', split_ratios=(0.8,0.1,0.1), seed=42):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        all_seqs = data['sequences']  # list of (tokens, quadrant)
        
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_seqs))
        n = len(all_seqs)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])
        
        if split == 'train':
            idx = idx[:n_train]
        elif split == 'val':
            idx = idx[n_train:n_train+n_val]
        else:
            idx = idx[n_train+n_val:]
        
        self.samples = [all_seqs[i] for i in idx]
        print(f"[Dataset] {split}: {len(self.samples)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, quadrant = self.samples[idx]
        
        # Truncate or pad to MAX_SEQ+1 (need +1 for input/target shift)
        tokens = tokens[:MAX_SEQ+1]
        if len(tokens) < MAX_SEQ+1:
            tokens = tokens + [PAD] * (MAX_SEQ+1 - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        x = tokens[:-1]   # input: [0..MAX_SEQ-1]
        y = tokens[1:]    # target: [1..MAX_SEQ]
        label = torch.tensor(quadrant, dtype=torch.long)
        return x, y, label
