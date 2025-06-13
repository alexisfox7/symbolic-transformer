# Different kinds of layer norms
import torch
import torch.nn as nn
from torch.nn import functional as F

class VanillaNorm(nn.Module):
    """Standard layer normalization."""
    #TODO: old code
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    #TODO: old code
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)
    
class ChannelNorm(nn.Module):
    """
    LayerNorm that preserves symbolic structure by operating on each head channel independently.
    This maintains the structured token space properties required for symbolic reasoning.
    """
    #TODO: old code
    def __init__(self, n_embd, n_head, bias=True):
        super().__init__()
        assert n_embd > 0, "n_embd must be positive"
        assert n_head > 0, "n_head must be positive"
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        
        # Separate normalization parameters for each head channel
        self.channel_weights = nn.Parameter(torch.ones(n_head))
        self.channel_biases = nn.Parameter(torch.zeros(n_head)) if bias else None
        
    #TODO: old code
    def forward(self, x):
        """Apply channel-wise layer normalization preserving head structure."""
        B, T, C = x.shape
        assert C == self.n_embd, f"Input embedding dimension {C} != expected {self.n_embd}"
        
        x_heads = x.view(B, T, self.n_head, self.head_dim) # Reshape to separate head channels: (B, T, n_head, head_dim)
        
        #NOTE could weight and bias be passed here (simplify)
        normalized = F.layer_norm(x_heads, (self.head_dim,), eps=1e-5) # Vectorized layer normalization across head_dim for all channels
         
        # Apply channel-specific weights and biases using broadcasting
        normalized = normalized * self.channel_weights[None, None, :, None]
        if self.channel_biases is not None:
            normalized = normalized + self.channel_biases[None, None, :, None]
        
        # Reshape back to original format: (B, T, n_embd)
        return normalized.view(B, T, C)
