"""
EmoMusicTransformer — GPT-style Transformer with:
1. Discrete emotion embedding injected at every layer via FiLM
2. Continuous valence/arousal conditioning
3. Emotion trajectory support (per-segment emotion labels)
"""
import torch
import torch.nn as nn
import math

VOCAB_SIZE  = 263
MAX_SEQ     = 512
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 6
D_FF        = 1024
DROPOUT     = 0.2
NUM_EMOTIONS= 4
EMO_DIM     = 64

# Russell circumplex coordinates per quadrant (valence, arousal) in [-1,1]
RUSSELL_COORDS = {
    0: ( 1.0,  1.0),   # Q1 Happy
    1: ( 1.0, -1.0),   # Q2 Relaxed
    2: (-1.0, -1.0),   # Q3 Sad
    3: (-1.0,  1.0),   # Q4 Tense
}

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for emotion conditioning."""
    def __init__(self, d_model, emo_dim):
        super().__init__()
        self.gamma = nn.Linear(emo_dim, d_model)
        self.beta  = nn.Linear(emo_dim, d_model)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, emb):
        # x: (B, T, D), emb: (B, emo_dim)
        g = self.gamma(emb).unsqueeze(1)  # (B,1,D)
        b = self.beta(emb).unsqueeze(1)   # (B,1,D)
        return g * x + b


class EmoTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, emo_dim):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.film    = FiLM(d_model, emo_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb, attn_mask=None):
        # Self-attention with causal mask
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = self.norm1(x + self.dropout(attn_out))
        # FiLM emotion conditioning
        x = self.film(x, emb)
        # Feed-forward
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class EmoMusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Token + positional embeddings
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos_emb   = nn.Embedding(MAX_SEQ, D_MODEL)
        
        # Discrete emotion embedding (trainable)
        self.emo_emb   = nn.Embedding(NUM_EMOTIONS, EMO_DIM)
        
        # Continuous valence/arousal projection
        self.va_proj   = nn.Linear(2, EMO_DIM)
        
        # Combined emotion MLP
        self.emo_mlp   = nn.Sequential(
            nn.Linear(EMO_DIM * 2, EMO_DIM),
            nn.GELU(),
            nn.Linear(EMO_DIM, EMO_DIM)
        )
        
        # Transformer layers with FiLM conditioning
        self.layers = nn.ModuleList([
            EmoTransformerLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, EMO_DIM)
            for _ in range(N_LAYERS)
        ])
        
        self.norm    = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Weight-tied output projection
        self.out_proj = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.out_proj.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_emotion_embedding(self, emotion_label):
        """
        Combine discrete embedding + continuous valence/arousal.
        emotion_label: (B,) long tensor
        Returns: (B, EMO_DIM)
        """
        # Discrete embedding
        discrete = self.emo_emb(emotion_label)  # (B, EMO_DIM)
        
        # Continuous Russell coordinates
        coords = torch.tensor(
            [RUSSELL_COORDS[e.item()] for e in emotion_label],
            dtype=torch.float32, device=emotion_label.device
        )  # (B, 2)
        continuous = self.va_proj(coords)  # (B, EMO_DIM)
        
        # Combine
        combined = self.emo_mlp(torch.cat([discrete, continuous], dim=-1))
        return combined

    def forward(self, x, emotion_label):
        """
        x: (B, T) token ids
        emotion_label: (B,) quadrant index
        """
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        
        # Token + position embedding
        h = self.dropout(self.token_emb(x) + self.pos_emb(pos))
        
        # Emotion embedding (discrete + continuous combined)
        emb = self.get_emotion_embedding(emotion_label)  # (B, EMO_DIM)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # Transformer layers — emotion injected at every layer via FiLM
        for layer in self.layers:
            h = layer(h, emb, attn_mask=mask)
        
        h = self.norm(h)
        logits = self.out_proj(h)  # (B, T, VOCAB_SIZE)
        return logits

    def generate(self, emotion_label, device, max_new_tokens=512,
                 temperature=1.0, top_p=0.9):
        """
        Autoregressive generation with nucleus sampling.
        emotion_label: int (0-3)
        Returns: list of token ids
        """
        BOS = 1
        self.eval()
        
        # Emotion-aware temperature
        emo_temps = {0: 1.1, 1: 0.8, 2: 0.9, 3: 1.2}  # Happy=adventurous, Relaxed=smooth
        temperature = emo_temps.get(emotion_label, temperature)
        
        label = torch.tensor([emotion_label], dtype=torch.long, device=device)
        tokens = torch.tensor([[BOS]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to MAX_SEQ
                inp = tokens[:, -MAX_SEQ:]
                logits = self.forward(inp, label)
                logits = logits[:, -1, :] / temperature
                
                # Nucleus sampling (top-p)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                # Remove tokens beyond top-p
                sorted_logits[cumprobs - probs > top_p] = float('-inf')
                probs = torch.softmax(sorted_logits, dim=-1)
                next_token = sorted_idx[0, torch.multinomial(probs[0], 1)]
                
                if next_token.item() == 2:  # EOS
                    break
                
                tokens = torch.cat([tokens, next_token.view(1,1)], dim=1)
        
        return tokens[0].tolist()

    def generate_trajectory(self, emotion_sequence, device,
                            tokens_per_segment=128, top_p=0.9):
        """
        NOVEL: Generate music with evolving emotion trajectory.
        emotion_sequence: list of emotion labels, e.g. [0, 2, 3] (Happy->Sad->Tense)
        tokens_per_segment: how many tokens per emotion segment
        Returns: full token sequence
        """
        self.eval()
        BOS = 1
        all_tokens = [BOS]
        context = torch.tensor([[BOS]], dtype=torch.long, device=device)
        
        emo_temps = {0: 1.1, 1: 0.8, 2: 0.9, 3: 1.2}
        
        with torch.no_grad():
            for seg_idx, emo in enumerate(emotion_sequence):
                label = torch.tensor([emo], dtype=torch.long, device=device)
                temperature = emo_temps[emo]
                seg_tokens = 0
                
                while seg_tokens < tokens_per_segment:
                    inp = context[:, -MAX_SEQ:]
                    logits = self.forward(inp, label)
                    logits = logits[:, -1, :] / temperature
                    
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    sorted_logits[cumprobs - probs > top_p] = float('-inf')
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_token = sorted_idx[0, torch.multinomial(probs[0], 1)]
                    
                    if next_token.item() == 2:  # EOS — don't stop mid-trajectory
                        break
                    context = torch.cat([context, next_token.view(1,1)], dim=1)


                    all_tokens.append(next_token.item())
                    seg_tokens += 1
        
        all_tokens.append(2)  # EOS
        return all_tokens
