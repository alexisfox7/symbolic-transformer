# Different kinds of attention
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#NOTE pretty sure this works, similar to karpathys
class VanillaAttention(nn.Module):
    """Standard causal self-attention mechanism."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # batched Q, K, V for all heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # output proj
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 

    def forward(self, x):
        B, T, C = x.size() # T = block_size, C = n_embd

        # calc Q, K, V for all heads in batch
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C)
        
        # reshape for multi-head attention: (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  

        # attention scores with scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale # (B, nh, hd, T) -> (B, nh, T, T)
        
        # causal mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # softmax, dropout, apply to values
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hd)
        
        # concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y

class SymbolicAttention(nn.Module):
    """
    Symbolic self-attention with ALiBi positional encoding and optional Kronecker-lifted matrices.
    """
 
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.use_v = getattr(config, 'use_v', False)
        self.use_proj = getattr(config, 'use_proj', False)

        if self.use_v:
            self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias) # only Q and K projections 
            self.v_tmp = nn.Parameter(torch.randn(config.n_head, config.n_head) * 0.02) # kronecker-lifted V matrix parameter
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # standard Q, K, V projections
        
        if self.use_proj:
            self.proj_tmp = nn.Parameter(torch.randn(config.n_head, config.n_head) * 0.02)
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # ALiBi slopes - computed once and cached
        slopes = self._get_alibi_slopes(config.n_head)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    #TODO: check if same as old function
    def _get_kronecker_lifted_tensor(self, v):
        """
        Lift head-to-head matrix to full embedding dimension using Kronecker product.
        Creates block-diagonal structure preserving head channels.
        """
        identity = torch.eye(self.head_dim, device=v.device, dtype=v.dtype)
        return torch.kron(v, identity)

    # havent tested but prob works
    def _get_alibi_slopes(self, n_heads):
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n_heads):
            start = 2**(-(2**-(math.log2(n_heads)-3)))
            ratio = start
            return [start*ratio**i for i in range(n_heads)]

        def get_slopes(n_heads):
            if n_heads <= 0:
                return []
            
            # Check if n_heads is a power of 2
            if (n_heads & (n_heads - 1)) == 0:
                return get_slopes_power_of_2(n_heads)
            else:
                # Handle non-power-of-2 case
                closest_power_of_2 = 2**math.floor(math.log2(n_heads))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                
                # Get additional slopes for remaining heads
                if n_heads > closest_power_of_2:
                    extra_base = 2**(-(2**-(math.log2(2*closest_power_of_2)-3)))
                    num_remaining = n_heads - closest_power_of_2
                    extra_slopes = [extra_base * (extra_base**i) for i in range(num_remaining)]
                    slopes.extend(extra_slopes)
                
                return slopes[:n_heads]

        slopes = get_slopes(n_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len, device):
        """Generate ALiBi bias matrix for the given sequence length."""
        # Create position indices
        context_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        memory_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute relative distances (memory - context)
        relative_position = memory_position[None, :] - context_position[:, None]
        
        # For causal attention, future positions should have large negative bias
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Apply slopes to relative positions
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]
        
        # Apply causal masking
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias

    def forward(self, x):
        """
        Forward pass with optional Kronecker-lifted V matrix.
        
        Args:
            x: Input symbolic state (B, T, n_embd)
        """
        B, T, C = x.size()

        if self.use_v:
            # Separate Q, K from input and compute V using Kronecker lifting
            qk = self.c_attn(x)
            q, k = qk.split(self.n_embd, dim=2)
            
            # Apply Kronecker-lifted V transformation
            v_matrix = self._get_kronecker_lifted_tensor(self.v_tmp)
            x_flat = x.view(-1, C)
            v = torch.matmul(x_flat, v_matrix).view(B, T, C) #REVIEW check if i need transpose
        else:
            # Standard Q, K, V projections
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores with proper scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        att_scores = (q @ k.transpose(-2, -1)) * scale

        # add ALiBi bias
        if T > 1:
            alibi_bias = self._get_alibi_bias(T, x.device)
            att_scores = att_scores + alibi_bias[None, :, :, :]

        # softmax, dropout, value
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)
        y = att_weights @ v  # (B, nh, T, hs)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        if self.use_proj:
            proj_matrix = self._get_kronecker_lifted_tensor(self.proj_tmp)
            y_flat = y.view(-1, C)
            y = torch.matmul(y_flat, proj_matrix.t()).view(B, T, C)
        else:
            y = self.c_proj(y)
        
        y = self.resid_dropout(y)
        
        return y