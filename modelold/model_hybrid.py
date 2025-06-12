# ./model/model_hybrid_symbolic_transformer.py
"""
Hybrid Symbolic Transformer model that combines symbolic attention with normal FFN.
Uses symbolic attention (SymbolicCausalSelfAttentionALiBi) but replaces the vocabulary-
constrained FFN with a standard feed-forward network for faster training and inference.

This provides a middle ground between full symbolic constraints and standard transformers,
maintaining symbolic attention benefits while avoiding FFN vocabulary projection overhead.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torch.serialization
from config import SymbolicConfig


class SymbolicLayerNorm(nn.Module):
    """
    LayerNorm that preserves symbolic structure by operating on each head channel independently.
    This maintains the structured token space properties required for symbolic reasoning.
    """
    def __init__(self, n_embd, n_head, bias=True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        
        # Separate normalization parameters for each head channel
        self.channel_weights = nn.Parameter(torch.ones(n_head))
        self.channel_biases = nn.Parameter(torch.zeros(n_head)) if bias else None
        
    def forward(self, x):
        """Apply channel-wise layer normalization preserving head structure."""
        B, T, C = x.shape
        assert C == self.n_embd, f"Input embedding dimension {C} != expected {self.n_embd}"
        
        # Reshape to separate head channels: (B, T, n_head, head_dim)
        x_heads = x.view(B, T, self.n_head, self.head_dim)
        
        # Apply layer norm to each head channel independently
        normalized_heads = torch.zeros_like(x_heads)
        for h in range(self.n_head):
            # Extract channel h: (B, T, head_dim)
            channel_data = x_heads[:, :, h, :]
            
            # Normalize across the head_dim dimension
            normalized_channel = F.layer_norm(
                channel_data, 
                (self.head_dim,), 
                eps=1e-5
            )
            
            # Scale by channel-specific weight and add channel-specific bias
            normalized_channel = normalized_channel * self.channel_weights[h]
            if self.channel_biases is not None:
                normalized_channel = normalized_channel + self.channel_biases[h]
                
            normalized_heads[:, :, h, :] = normalized_channel
        
        # Reshape back to original format: (B, T, C)
        return normalized_heads.view(B, T, C)


class StandardFeedForward(nn.Module):
    """Standard feed-forward network without vocabulary constraints."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SymbolicCausalSelfAttentionALiBi(nn.Module):
    """
    Symbolic self-attention with ALiBi positional encoding and optional Kronecker-lifted V matrix.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_v = getattr(config, 'use_v', True)
        self.use_proj = getattr(config, 'use_proj', True)
        
        # Q and K projections (always standard)
        if self.use_v:
            self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Kronecker-lifted V transformation parameters
        if self.use_v:
            self.v_tmp = nn.Parameter(torch.randn(config.n_head, self.head_dim, self.head_dim))
        
        # Output projection
        if self.use_proj:
            self.proj_tmp = nn.Parameter(torch.randn(config.n_head, self.head_dim, self.head_dim))
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # ALiBi slopes
        self.register_buffer('alibi_slopes', self._get_alibi_slopes(config.n_head))
        
        # Vocab embeddings reference (set externally)
        self.vocab_embeddings_ref = None
        
    def _get_alibi_slopes(self, n_heads):
        """Generate ALiBi slopes for attention bias."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2])
            return torch.tensor(slopes)
    
    def _get_alibi_bias(self, seq_len, device):
        """Generate ALiBi positional bias matrix."""
        # Create position differences matrix
        positions = torch.arange(seq_len, device=device)[None, :] - torch.arange(seq_len, device=device)[:, None]
        positions = positions.clamp(max=0)  # Only look at past positions
        
        # Apply slopes to create bias
        bias = positions[None, None, :, :] * self.alibi_slopes[:, None, None].to(device)
        return bias
    
    def _get_kronecker_lifted_tensor(self, tmp_tensor):
        """Convert per-head parameters to full Kronecker-lifted tensor."""
        n_head, head_dim, _ = tmp_tensor.shape
        I_head = torch.eye(head_dim, device=tmp_tensor.device, dtype=tmp_tensor.dtype)
        
        kronecker_blocks = []
        for h in range(n_head):
            # Create Kronecker product: I ⊗ tmp_tensor[h]
            kron_block = torch.kron(I_head, tmp_tensor[h])
            kronecker_blocks.append(kron_block)
        
        # Block diagonal matrix
        return torch.block_diag(*kronecker_blocks)
    
    def forward(self, x):
        """
        Forward pass for symbolic attention.
        
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
            v = torch.matmul(x_flat, v_matrix.t()).view(B, T, C)
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

        # Add ALiBi bias
        if T > 1:
            alibi_bias = self._get_alibi_bias(T, x.device)
            att_scores = att_scores + alibi_bias[None, :, :, :]

        # Apply softmax and dropout
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        # Apply attention to values
        y = att_weights @ v  # (B, nh, T, hs)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection
        if self.use_proj:
            # Use structured Kronecker-lifted projection
            proj_matrix = self._get_kronecker_lifted_tensor(self.proj_tmp)
            y_flat = y.view(-1, C)
            y = torch.matmul(y_flat, proj_matrix.t()).view(B, T, C)
        else:
            # Standard linear projection
            y = self.c_proj(y)
        
        y = self.resid_dropout(y)
        
        return y


class HybridSymbolicTransformerBlock(nn.Module):
    """
    Hybrid transformer block that uses symbolic attention with standard FFN.
    Combines the benefits of symbolic attention with the efficiency of normal FFNs.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Symbolic layer normalization (preserves head structure)
        self.ln_1 = SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_2 = SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias)

        # Symbolic attention mechanism
        self.attn = SymbolicCausalSelfAttentionALiBi(config)

        # Standard FFN (no vocabulary constraints)
        self.ffn = StandardFeedForward(config)

    def forward(self, x):
        """
        Forward pass for hybrid transformer block.

        Args:
            x: Input state (B, T, n_embd)

        Returns:
            Updated state (B, T, n_embd)
        """
        # Symbolic attention path
        norm_for_attn = self.ln_1(x)
        attn_output = self.attn(norm_for_attn)
        x = x + attn_output

        # Standard FFN path
        norm_for_ffn = self.ln_2(x)
        ffn_output = self.ffn(norm_for_ffn)
        x = x + ffn_output

        return x


class HybridSymbolicTransformerModel(nn.Module):
    """
    Hybrid Symbolic Transformer model that uses symbolic attention with normal FFN.
    This provides interpretable attention mechanisms while maintaining computational efficiency
    in the feed-forward layers.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)
        
        # Core model components (no positional embeddings with ALiBi)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([HybridSymbolicTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias),
        ))
        
        # Pass vocabulary embedding reference to all blocks after creation
        for block in self.transformer.h:
            block.attn.vocab_embeddings_ref = self.transformer.wte

        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight 

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for symbolic components
        for pn, p in self.named_parameters():
            if 'v_tmp' in pn or 'proj_tmp' in pn:
                # Initialize Kronecker parameters
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

        print(f"HybridSymbolicTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Architecture: Symbolic attention + Standard FFN")
        print(f"Vocabulary size: {config.vocab_size}, Embedding dim: {config.n_embd}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the hybrid symbolic transformer.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation (B, T), optional
            
        Returns:
            If targets is None: logits (B, T, vocab_size)
            If targets is provided: (logits, loss)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token embeddings (no position embeddings due to ALiBi)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference mode: only return logits
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # If sequence is too long, crop it
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizers with different weight decay for different parameter types.
        """
        # Start with all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


# Helper function to create model from config
def create_hybrid_symbolic_model(config):
    """Create a HybridSymbolicTransformerModel from a config."""
    return HybridSymbolicTransformerModel(config)