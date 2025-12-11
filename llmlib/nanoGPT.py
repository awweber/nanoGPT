import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameter
n_embd = 384     # size of the embedding vectors (dimension)
n_head = 6       # number of attention heads (384 / 6 = 64 dim per head)
n_layer = 6      # number of transformer blocks
block_size = 256 # Context: The model looks back 256 characters
vocab_size = 65  # number of unique characters in the vocabulary
dropout = 0.2    # against overfitting

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class Head(nn.Module):
    """ One attention head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register buffer, so that it is not a trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower triangular matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # type: ignore # Masking future tokens
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple Heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Output der Heads konkatenieren
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Ein einfaches lineares Layer gefolgt von Nicht-Linearität """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expansion (Standard in Transformern)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection zurück
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Kommunikation gefolgt von Berechnung """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # Communication
        self.ffwd = FeedFoward(n_embd)                  # Computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual Connections (x + ...) sind extrem wichtig für tiefes Lernen!
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """ The GPT Language Model """
    def __init__(self):
        super().__init__()
        # Embedding für Token-Identität
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Embedding für Position im Satz (WICHTIG!)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Die Transformer Blöcke
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Finaler Layer Norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # Projektion auf Vokabular

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token Emb + Positional Emb
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # Durch die Transformer Blöcke
        x = self.blocks(x) 
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Kontext beschneiden, wenn er zu lang wird (wir haben max block_size PosEmbeddings)
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx