# ./model/model_symbolic_standard_norm.py
"""
Symbolic Transformer with Standard LayerNorm (Test Version)
This version replaces channel-wise LayerNorm with standard LayerNorm to test
the impact of channel structure preservation on symbolic interpretability.

The ONLY difference from the main Symbolic Transformer is:
- Uses standard nn.LayerNorm instead of SymbolicLayerNorm
- All other symbolic constraints remain identical
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torch.serialization
from config import SymbolicConfig 


class VocabularyProjectionFFN(nn.Module):
    """
    Feed Forward Network (FFN) that constrains outputs to the vocabulary embedding manifold
    with proper head-channel decomposition and TRUE weight tying to embedding matrix.
    Each head channel operates independently on its corresponding slice of the vocabulary embeddings.
    
    IDENTICAL to main symbolic version - no changes needed here.
    """
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        assert config.vocab_size > 0, "vocab_size must be positive"
        assert config.n_embd > 0, "n_embd must be positive"
        assert config.n_head > 0, "n_head must be positive"
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)

        # Store reference to vocabulary embeddings (not a copy)
        self.vocab_embeddings_ref = vocab_embeddings_ref

        # Batched FFN transformations for all channels
        self.channel_ffns = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)

        # Per-channel learnable temperature for attention sharpness
        self.channel_temperatures = nn.Parameter(torch.ones(self.n_head))

        # Optional refinement layers per channel
        self.use_refinement = getattr(config, 'use_vocab_refinement', False)
        if self.use_refinement:
            self.channel_refinements = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
            self.channel_refinement_gates = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)

    def _get_vocab_channels_batched(self):
        """Extract all channel slices of vocabulary embeddings efficiently."""
        vocab_reshaped = self.vocab_embeddings_ref.weight.view(
            self.vocab_size, self.n_head, self.head_dim
        ).transpose(0, 1)
        return vocab_reshaped  # (n_head, vocab_size, head_dim)

    def forward(self, x):
        """
        Project input to vocabulary embedding manifold via channel-wise TRUE vocabulary constraint.
        Uses direct similarity computation with embedding weights - no additional parameters.
        """
        B, T, C = x.shape
        assert C == self.n_embd, f"Input dim {C} != expected {self.n_embd}"

        # Reshape to separate and flatten head channels
        x_channels = x.view(B, T, self.n_head, self.head_dim)
        x_flat = x_channels.view(B * T * self.n_head, self.head_dim)

        # Apply batched FFN transformation to all channels simultaneously
        ffn_output = self.channel_ffns(x_flat)
        
        # Reshape FFN output for batched vocabulary similarity computation
        ffn_reshaped = ffn_output.view(B * T, self.n_head, self.head_dim)

        # Get all vocabulary channel embeddings efficiently
        vocab_channels = self._get_vocab_channels_batched()

        # Compute TRUE vocabulary attention using direct similarity
        vocab_logits = torch.einsum('bnh,hvd->bnv', ffn_reshaped, vocab_channels)
        
        # Apply channel-specific temperature scaling
        temps_clamped = torch.clamp(self.channel_temperatures, min=0.1)[None, :, None]
        vocab_logits_scaled = vocab_logits / temps_clamped
        
        # Compute attention weights
        vocab_weights = F.softmax(vocab_logits_scaled, dim=-1)

        # Batched vocabulary projection
        vocab_output = torch.einsum('bnv,hvd->bnh', vocab_weights, vocab_channels)

        # Optional refinement while maintaining vocabulary grounding
        if self.use_refinement:
            x_flat_for_refinement = x_channels.view(B * T * self.n_head, self.head_dim)
            vocab_flat = vocab_output.view(B * T * self.n_head, self.head_dim)
            
            refinement = torch.tanh(self.channel_refinements(x_flat_for_refinement))
            gate = torch.sigmoid(self.channel_refinement_gates(x_flat_for_refinement))
            
            refined_output = vocab_flat * (1 - gate) + refinement * gate
            
            # Re-project to ensure vocabulary grounding is maintained
            refined_reshaped = refined_output.view(B * T, self.n_head, self.head_dim)
            refined_logits = torch.einsum('bnh,hvd->bnv', refined_reshaped, vocab_channels)
            refined_logits_scaled = refined_logits / temps_clamped
            refined_weights = F.softmax(refined_logits_scaled, dim=-1)
            
            vocab_output = torch.einsum('bnv,hvd->bnh', refined_weights, vocab_channels)

        # Reshape back to original format
        output = vocab_output.view(B, T, self.n_embd)

        return self.dropout(output)


class SymbolicCausalSelfAttentionALiBi(nn.Module):
    """
    Causal self-attention mechanism for the Symbolic Transformer with ALiBi positional encoding.
    IDENTICAL to main symbolic version - no changes needed here.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd > 0, "n_embd must be positive"
        assert config.n_head > 0, "n_head must be positive"
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Standard Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Optional structured output projection using Kronecker lifting
        self.use_proj = getattr(config, 'use_proj', False)
        if self.use_proj:
            self.proj_tmp = nn.Parameter(torch.randn(config.n_head, config.n_head) * 0.02)
        else:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # ALiBi slopes - computed once and cached
        slopes = self._get_alibi_slopes(config.n_head)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def _get_kronecker_lifted_tensor(self, v):
        """Lift head-to-head matrix to full embedding dimension using Kronecker product structure."""
        n_heads = v.shape[0]
        head_dim = self.n_embd // n_heads
        
        v_out = torch.zeros(self.n_embd, self.n_embd, device=v.device, dtype=v.dtype)
        
        for i in range(n_heads):
            for j in range(n_heads):
                start_i, end_i = i * head_dim, (i + 1) * head_dim
                start_j, end_j = j * head_dim, (j + 1) * head_dim
                v_out[start_i:end_i, start_j:end_j] = v[i, j] * torch.eye(head_dim, device=v.device, dtype=v.dtype)
        
        return v_out

    def _get_alibi_slopes(self, n_heads):
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n_heads):
            start = 2**(-(2**-(math.log2(n_heads)-3)))
            ratio = start
            return [start*ratio**i for i in range(n_heads)]

        def get_slopes(n_heads):
            if n_heads <= 0:
                return []
            
            if (n_heads & (n_heads - 1)) == 0:
                return get_slopes_power_of_2(n_heads)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n_heads))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                
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
        context_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        memory_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        relative_position = memory_position[None, :] - context_position[:, None]
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias

    def forward(self, x):
        """Forward pass for symbolic attention."""
        B, T, C = x.size()

        # Calculate query, key, and value from input
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

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
        y = att_weights @ v

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection
        if self.use_proj:
            proj_matrix = self._get_kronecker_lifted_tensor(self.proj_tmp)
            y_flat = y.view(-1, C)
            y = torch.matmul(y_flat, proj_matrix.t()).view(B, T, C)
        else:
            y = self.c_proj(y)
        
        y = self.resid_dropout(y)
        
        return y


class SymbolicTransformerBlock(nn.Module):
    """
    Symbolic transformer block using STANDARD LayerNorm instead of channel-wise.
    This is the KEY DIFFERENCE for testing channel structure impact.
    """
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.config = config

        # CHANGE: Use standard LayerNorm instead of SymbolicLayerNorm
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Symbolic attention mechanism (no vocab reference needed)
        self.attn = SymbolicCausalSelfAttentionALiBi(config)

        # Optional vocabulary-constrained FFN
        self.use_symbolic_ffn = getattr(config, 'use_symbolic_ffn', True)
        if self.use_symbolic_ffn:
            self.ffn = VocabularyProjectionFFN(config, vocab_embeddings_ref)

    def forward(self, xt):
        """
        Forward pass for symbolic transformer block with STANDARD layer normalization.
        """
        # Symbolic attention path with STANDARD LayerNorm
        norm_for_attn = self.ln_1(xt)
        attn_output = self.attn(norm_for_attn)
        xt = xt + attn_output

        # Optional symbolic FFN path with STANDARD LayerNorm
        if self.use_symbolic_ffn:
            norm_for_ffn = self.ln_2(xt)
            ffn_output = self.ffn(norm_for_ffn)
            xt = xt + ffn_output

        return xt


class SymbolicTransformerStandardNorm(nn.Module):
    """
    Symbolic Transformer model using STANDARD LayerNorm instead of channel-wise LayerNorm.
    
    This version tests the impact of channel structure preservation on symbolic interpretability.
    All other symbolic constraints remain identical to the main Symbolic Transformer.
    
    KEY DIFFERENCE: Uses nn.LayerNorm instead of SymbolicLayerNorm throughout.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        assert config.vocab_size > 0, "vocab_size must be positive"
        assert config.n_embd > 0, "n_embd must be positive"
        assert config.n_head > 0, "n_head must be positive"
        assert config.n_layer > 0, "n_layer must be positive"
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)
        
        # Store configuration flags
        self.use_symbolic_ffn = getattr(config, 'use_symbolic_ffn', True)
        self.use_proj = getattr(config, 'use_proj', False)
        self.use_vocab_refinement = getattr(config, 'use_vocab_refinement', False)
        
        # Core model components (no positional embeddings with ALiBi)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                SymbolicTransformerBlock(config, None) for _ in range(config.n_layer)
            ]),
            # CHANGE: Use standard LayerNorm for final layer norm
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Pass vocabulary embedding reference to all blocks after creation
        for block in self.transformer.h:
            if hasattr(block, 'ffn') and block.ffn is not None:
                block.ffn.vocab_embeddings_ref = self.transformer.wte

        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Vocabulary grounding layer for final output
        self.vocab_grounding = VocabularyProjectionFFN(config, self.transformer.wte)

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for symbolic components
        for pn, p in self.named_parameters():
            if 'vocab_attention' in pn and 'weight' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.01)
            elif pn.endswith('temperature'):
                torch.nn.init.constant_(p, 1.0)
            elif pn.endswith('c_proj.weight') or 'proj_tmp' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"SymbolicTransformerStandardNorm initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"KEY DIFFERENCE: Uses STANDARD LayerNorm instead of channel-wise LayerNorm")
        print(f"Symbolic constraints: symbolic_ffn={self.use_symbolic_ffn}, vocab_refinement={self.use_vocab_refinement}")
        print(f"Architecture: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wte'):
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize model weights with symbolic-aware initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, SymbolicCausalSelfAttentionALiBi):
            if hasattr(module, 'proj_tmp'):
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.LayerNorm):
            # Standard LayerNorm initialization
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the SymbolicTransformerStandardNorm.
        IDENTICAL to main symbolic version except for LayerNorm usage.
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Robust input validation
        if not (1 <= t <= getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)):
            max_len = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
            raise ValueError(f"Sequence length {t} must be between 1 and {max_len}")
        
        if torch.any(input_ids >= self.config.vocab_size) or torch.any(input_ids < 0):
            raise ValueError(f"Input IDs must be in range [0, {self.config.vocab_size})")

        # Token embeddings only (no positional embeddings with ALiBi)
        tok_emb = self.transformer.wte(input_ids)
        
        # Initialize symbolic stream
        xt = self.transformer.drop(tok_emb)

        # Pass through symbolic transformer blocks
        for block in self.transformer.h:
            xt = block(xt)

        # Final vocabulary grounding and normalization
        xt_grounded = self.vocab_grounding(xt)
        x_final = self.transformer.ln_f(xt_grounded)
        logits = self.lm_head(x_final)

        # Calculate language modeling loss if labels provided
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(f"Labels shape {labels.shape} must match input_ids shape {input_ids.shape}")
            
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive if specified")
        
        self.eval()
        
        max_total_length = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_total_length else idx[:, -max_total_length:]
            
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            logits = logits[:, -1, :] / max(temperature, 1e-7)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            
            if temperature > 0:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx

    def get_model_info(self):
        """Return comprehensive model information."""
        info = {
            'model_class': self.__class__.__name__,
            'key_difference': 'Uses standard LayerNorm instead of channel-wise LayerNorm',
            'total_parameters': self.get_num_params(),
            'non_embedding_parameters': self.get_num_params(non_embedding=True),
            'configuration': {
                'vocab_size': self.config.vocab_size,
                'n_embd': self.config.n_embd,
                'n_head': self.config.n_head,
                'n_layer': self.config.n_layer,
                'block_size': self.config.block_size,
                'dropout': self.config.dropout,
            },
            'symbolic_features': {
                'use_symbolic_ffn': self.use_symbolic_ffn,
                'use_proj': self.use_proj,
                'use_vocab_refinement': self.use_vocab_refinement,
                'layernorm_type': 'standard',  # KEY DIFFERENCE
            },
            'architecture_details': {
                'head_dim': self.config.n_embd // self.config.n_head,
                'padding_idx': self.padding_idx,
                'max_position_embeddings': getattr(self.config, 'max_position_embeddings', self.config.block_size * 4),
            }
        }
        
        if len(list(self.parameters())) > 0:
            info['device'] = str(next(self.parameters()).device)
            info['dtype'] = str(next(self.parameters()).dtype)
        
        return info


# Export the test model class
__all__ = ['SymbolicTransformerStandardNorm']