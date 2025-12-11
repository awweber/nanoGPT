import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameter ---
batch_size = 32 # How many independent sequences to process in parallel
block_size = 8  # Maximum length of context (irrelevant for Bigram, but important for later)
vocab_size = 65 # Size of vocabulary

# device configuration
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # M4 Check!

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        # Embedding Dimension = Vocab Size, since we have no Hidden Layers
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            # Reshape for loss computation
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # (B,C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    