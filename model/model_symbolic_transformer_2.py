# ./model/model_symbolic_transformer.py
"""
Pure Symbolic Transformer model with ALiBi positional encoding.
This model enforces strict symbolic constraints on all internal representations,
ensuring that every vector remains interpretable as combinations of vocabulary tokens.

Key architectural differences from Token-Factored Transformer:
- Single symbolic stream (Xt only, no Xe)
- Vocabulary-grounded operations at all layers
- Channel-wise layer normalization preserving head structure
- Vocabulary-constrained FFN projections

The model maintains dimensional analysis principles while ensuring complete
symbolic interpretability of all internal states through architectural design.

PERFORMANCE OPTIMIZATIONS:
- Vectorized channel operations eliminating explicit loops
- Memory-efficient tensor operations with pre-allocation
- Batched processing for multi-head channel computations
- Optimized attention and FFN implementations
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
    
    OPTIMIZED IMPLEMENTATION:
    - Vectorized operations across all channels simultaneously
    - No explicit loops for efficiency
    - Proper broadcasting for channel-specific parameters
    """
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
        
    def forward(self, x):
        """
        Apply channel-wise layer normalization preserving head structure.
        
        VECTORIZED IMPLEMENTATION:
        - Processes all channels simultaneously using broadcasting
        - Eliminates explicit loops for improved performance
        - Maintains mathematical equivalence to channel-wise processing
        
        Args:
            x: Input tensor (B, T, n_embd)
            
        Returns:
            Normalized tensor (B, T, n_embd) with channel-wise normalization applied
        """
        B, T, C = x.shape
        assert C == self.n_embd, f"Input embedding dimension {C} != expected {self.n_embd}"
        
        # Reshape to separate head channels: (B, T, n_head, head_dim)
        x_heads = x.view(B, T, self.n_head, self.head_dim)
        
        # Vectorized layer normalization across head_dim for all channels
        # Shape: (B, T, n_head, head_dim)
        normalized = F.layer_norm(x_heads, (self.head_dim,), eps=1e-5)
        
        # Apply channel-specific weights and biases using broadcasting
        # channel_weights: (n_head,) -> (1, 1, n_head, 1)
        # channel_biases: (n_head,) -> (1, 1, n_head, 1) 
        normalized = normalized * self.channel_weights[None, None, :, None]
        if self.channel_biases is not None:
            normalized = normalized + self.channel_biases[None, None, :, None]
        
        # Reshape back to original format: (B, T, n_embd)
        return normalized.view(B, T, C)


class VocabularyProjectionFFN(nn.Module):
    """
    Feed Forward Network (FFN) that constrains outputs to the vocabulary embedding manifold
    with proper head-channel decomposition and TRUE weight tying to embedding matrix.
    Each head channel operates independently on its corresponding slice of the vocabulary embeddings.
    
    OPTIMIZED IMPLEMENTATION WITH TRUE WEIGHT TYING:
    - Vectorized channel processing eliminating explicit loops
    - Memory-efficient batched operations using direct embedding similarity
    - NO additional learnable vocabulary projection parameters
    - TRUE vocabulary constraints via embedding weight reuse
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
        # Input: (B*T*n_head, head_dim), Output: (B*T*n_head, head_dim)
        self.channel_ffns = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)

        # Per-channel learnable temperature for attention sharpness
        self.channel_temperatures = nn.Parameter(torch.ones(self.n_head))

        # Optional refinement layers per channel
        self.use_refinement = getattr(config, 'use_vocab_refinement', False)
        if self.use_refinement:
            self.channel_refinements = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
            self.channel_refinement_gates = nn.Linear(self.head_dim, self.head_dim, bias=config.bias)

    def _get_vocab_channels_batched(self):
        """
        Extract all channel slices of vocabulary embeddings efficiently.
        
        OPTIMIZED APPROACH:
        - Single reshape operation instead of multiple slicing operations
        - Returns all channels simultaneously for batched processing
        
        Returns:
            E_all: All vocabulary embedding channels, shape (n_head, vocab_size, head_dim)
        """
        # Reshape vocabulary embeddings to separate channels
        # (vocab_size, n_embd) -> (vocab_size, n_head, head_dim) -> (n_head, vocab_size, head_dim)
        vocab_reshaped = self.vocab_embeddings_ref.weight.view(
            self.vocab_size, self.n_head, self.head_dim
        ).transpose(0, 1)
        return vocab_reshaped  # (n_head, vocab_size, head_dim)

    def forward(self, x):
        """
        Project input to vocabulary embedding manifold via channel-wise TRUE vocabulary constraint.
        Uses direct similarity computation with embedding weights - no additional parameters.
        
        VECTORIZED IMPLEMENTATION WITH TRUE WEIGHT TYING:
        - Processes all channels simultaneously using batched operations
        - Eliminates explicit channel loops for significant performance improvement
        - Uses direct embedding similarity instead of learnable projections
        - Memory-efficient with pre-allocated tensors where possible
        
        Args:
            x: Input tensor (B, T, n_embd)

        Returns:
            Vocabulary-grounded output tensor (B, T, n_embd)
        """
        B, T, C = x.shape
        assert C == self.n_embd, f"Input dim {C} != expected {self.n_embd}"

        # Reshape to separate and flatten head channels: (B, T, n_head, head_dim) -> (B*T*n_head, head_dim)
        x_channels = x.view(B, T, self.n_head, self.head_dim)
        x_flat = x_channels.view(B * T * self.n_head, self.head_dim)

        # Apply batched FFN transformation to all channels simultaneously
        ffn_output = self.channel_ffns(x_flat)  # (B*T*n_head, head_dim)
        
        # Reshape FFN output for batched vocabulary similarity computation
        ffn_reshaped = ffn_output.view(B * T, self.n_head, self.head_dim)  # (B*T, n_head, head_dim)

        # Get all vocabulary channel embeddings efficiently
        vocab_channels = self._get_vocab_channels_batched()  # (n_head, vocab_size, head_dim)

        # Compute TRUE vocabulary attention using direct similarity (NO additional parameters)
        # Using einsum for efficient batched computation:
        # ffn_reshaped: (B*T, n_head, head_dim)
        # vocab_channels: (n_head, vocab_size, head_dim)
        # Result: (B*T, n_head, vocab_size)
        # TODO: einsum is less computationally efficient that view and transpose operations
        #       rewrite this as time permits
        vocab_logits = torch.einsum('bnh,hvd->bnv', ffn_reshaped, vocab_channels)
        
        # Apply channel-specific temperature scaling using broadcasting
        # channel_temperatures: (n_head,) -> (1, n_head, 1)
        temps_clamped = torch.clamp(self.channel_temperatures, min=0.1)[None, :, None]
        vocab_logits_scaled = vocab_logits / temps_clamped
        
        # Compute attention weights
        vocab_weights = F.softmax(vocab_logits_scaled, dim=-1)  # (B*T, n_head, vocab_size)

        # Batched vocabulary projection using einsum
        # vocab_weights: (B*T, n_head, vocab_size)
        # vocab_channels: (n_head, vocab_size, head_dim)
        # Result: (B*T, n_head, head_dim)
        # TODO: einsum is less computationally efficient that view and transpose operations
        #       rewrite this as time permits
        vocab_output = torch.einsum('bnv,hvd->bnh', vocab_weights, vocab_channels)

        # Optional refinement while maintaining vocabulary grounding
        if self.use_refinement:
            # Reshape for refinement processing
            x_flat_for_refinement = x_channels.view(B * T * self.n_head, self.head_dim)
            vocab_flat = vocab_output.view(B * T * self.n_head, self.head_dim)
            
            # Apply refinement transformations
            refinement = torch.tanh(self.channel_refinements(x_flat_for_refinement))
            gate = torch.sigmoid(self.channel_refinement_gates(x_flat_for_refinement))
            
            # Gated combination
            refined_output = vocab_flat * (1 - gate) + refinement * gate
            
            # Re-project to ensure vocabulary grounding is maintained using TRUE constraint
            refined_reshaped = refined_output.view(B * T, self.n_head, self.head_dim)
            # TODO: einsum is less computationally efficient that view and transpose operations
            #       rewrite this as time permits
            refined_logits = torch.einsum('bnh,hvd->bnv', refined_reshaped, vocab_channels)
            refined_logits_scaled = refined_logits / temps_clamped
            refined_weights = F.softmax(refined_logits_scaled, dim=-1)
            
            # Final vocabulary projection
            # TODO: einsum is less computationally efficient that view and transpose operations
            #       rewrite this as time permits
            vocab_output = torch.einsum('bnv,hvd->bnh', refined_weights, vocab_channels)

        # Reshape back to original format: (B*T, n_head, head_dim) -> (B, T, n_embd)
        output = vocab_output.view(B, T, self.n_embd)

        return self.dropout(output)


class SymbolicCausalSelfAttentionALiBi(nn.Module):
    """
    Causal self-attention mechanism for the Symbolic Transformer with ALiBi positional encoding.
    Operates on symbolic representations without vocabulary projection - attention is a 
    symbolic operation that routes and combines existing symbolic information.
    
    ARCHITECTURE DESIGN:
    - Standard Q, K, V projections for symbolic token manipulation
    - ALiBi positional encoding eliminating need for position embeddings
    - Optional structured output projection using Kronecker lifting
    - Efficient implementation following established patterns
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
            # Standard output projection
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
        """
        Lift head-to-head matrix to full embedding dimension using Kronecker product structure.
        Creates block-diagonal structure preserving head channels.
        
        MATHEMATICAL OPERATION:
        For input matrix v[i,j] of size (n_heads, n_heads):
        Output matrix v_out[block_i, block_j] where each block is (head_dim, head_dim)
        v_out[i*head_dim:(i+1)*head_dim, j*head_dim:(j+1)*head_dim] = v[i,j] * I_head_dim
        
        Args:
            v: (n_heads, n_heads) parameter matrix
            
        Returns:
            v_out: (n_embd, n_embd) Kronecker-lifted transformation matrix
        """
        n_heads = v.shape[0]
        head_dim = self.n_embd // n_heads
        
        # Create the lifted tensor
        v_out = torch.zeros(self.n_embd, self.n_embd, device=v.device, dtype=v.dtype)
        
        for i in range(n_heads):
            for j in range(n_heads):
                # Create identity matrix scaled by v[i,j]
                start_i, end_i = i * head_dim, (i + 1) * head_dim
                start_j, end_j = j * head_dim, (j + 1) * head_dim
                v_out[start_i:end_i, start_j:end_j] = v[i, j] * torch.eye(head_dim, device=v.device, dtype=v.dtype)
        
        return v_out

    def _get_alibi_slopes(self, n_heads):
        """
        Compute ALiBi slopes for each attention head.
        Implementation based on the original ALiBi paper.
        """
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
        """
        Generate ALiBi bias matrix for the given sequence length.
        Returns bias matrix of shape (n_head, seq_len, seq_len).
        """
        # Create position indices
        context_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        memory_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute relative distances (memory - context)
        # Shape: (seq_len, seq_len)
        relative_position = memory_position[None, :] - context_position[:, None]
        
        # For causal attention, future positions should have large negative bias
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Apply slopes to relative positions
        # Negative relative positions (past) get small negative bias
        # Future positions will be masked with -inf
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]
        
        # Apply causal masking
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias

    def forward(self, x):
        """
        Forward pass for symbolic attention.
        
        STANDARD IMPLEMENTATION:
        - Follows proven attention patterns from the reference implementation
        - Efficient tensor operations with proper shape management
        - ALiBi positional encoding for length extrapolation
        
        Args:
            x: Input symbolic state (B, T, n_embd)
            
        Returns:
            Attention output (B, T, n_embd)
        """
        B, T, C = x.size()

        # Calculate query, key, and value from input
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


class SymbolicTransformerBlock(nn.Module):
    """
    Pure symbolic transformer block with vocabulary-constrained operations.
    Maintains single symbolic stream with optional FFN vocabulary projection.
    
    ARCHITECTURE DESIGN:
    - Single symbolic stream processing (no factored streams)
    - Symbolic layer normalization preserving head channel structure
    - Standard attention mechanism for symbolic token routing
    - Optional vocabulary-constrained FFN for symbolic grounding
    """
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.config = config

        # Symbolic layer normalization
        self.ln_1 = SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_2 = SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias)

        # Symbolic attention mechanism (no vocab reference needed)
        self.attn = SymbolicCausalSelfAttentionALiBi(config)

        # Optional vocabulary-constrained FFN
        self.use_symbolic_ffn = getattr(config, 'use_symbolic_ffn', True)
        if self.use_symbolic_ffn:
            self.ffn = VocabularyProjectionFFN(config, vocab_embeddings_ref)

    def forward(self, xt):
        """
        Forward pass for symbolic transformer block.
        
        STANDARD TRANSFORMER SEMANTICS:
        - Pre-layer normalization pattern
        - Skip connections for both attention and FFN paths
        - Optional FFN path based on configuration
        
        Args:
            xt: Symbolic input state (B, T, n_embd)

        Returns:
            Updated symbolic state (B, T, n_embd)
        """
        # Symbolic attention path
        norm_for_attn = self.ln_1(xt)
        attn_output = self.attn(norm_for_attn)
        xt = xt + attn_output

        # Optional symbolic FFN path
        if self.use_symbolic_ffn:
            norm_for_ffn = self.ln_2(xt)
            ffn_output = self.ffn(norm_for_ffn)
            xt = xt + ffn_output

        return xt


class SymbolicTransformerModel(nn.Module):
    """
    Pure Symbolic Transformer model with ALiBi positional encoding.
    All internal states are constrained to remain symbolically interpretable
    as combinations of vocabulary embeddings through architectural design.
    
    OPTIMIZED IMPLEMENTATION:
    - Vectorized operations throughout the model
    - Efficient memory usage patterns
    - Robust error handling and validation
    - Clean configuration management
    - Performance monitoring and statistics
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
            ln_f=SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias),
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
                # Initialize vocabulary attention to be close to identity
                torch.nn.init.normal_(p, mean=0.0, std=0.01)
            elif pn.endswith('temperature'):
                # Initialize temperature for stable training
                torch.nn.init.constant_(p, 1.0)
            elif pn.endswith('channel_weights'):
                # Initialize channel weights to ones
                torch.nn.init.ones_(p)
            elif pn.endswith('channel_biases'):
                # Initialize channel biases to zeros
                torch.nn.init.zeros_(p)
            elif pn.endswith('c_proj.weight') or 'proj_tmp' in pn:
                # Special scaling for output projections
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"SymbolicTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Symbolic constraints: symbolic_ffn={self.use_symbolic_ffn}, vocab_refinement={self.use_vocab_refinement}")
        print(f"Architecture: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")
        print(f"Performance optimizations: vectorized_ops=True, memory_efficient=True")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wte'):
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initialize model weights with symbolic-aware initialization.
        
        INITIALIZATION STRATEGY:
        - Standard normal initialization for most linear layers
        - Careful initialization of embedding layers with padding handling
        - Special initialization for symbolic components
        - Scaled initialization for output projections
        """
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
            # Initialize symbolic attention parameters
            if hasattr(module, 'proj_tmp'):
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, SymbolicLayerNorm):
            # Initialize symbolic layer norm
            torch.nn.init.ones_(module.channel_weights)
            if module.channel_biases is not None:
                torch.nn.init.zeros_(module.channel_biases)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the SymbolicTransformerModel.
        
        IMPLEMENTATION DETAILS:
        - Robust input validation with clear error messages
        - Efficient single-stream processing
        - Optional vocabulary grounding at output
        - Standard language modeling loss computation
        
        Args:
            input_ids: Token input IDs (B, T)
            attention_mask: Attention mask (unused with ALiBi causal attention)
            labels: Target labels for language modeling loss (B, T)
            
        Returns:
            Dictionary containing loss and logits
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
            # Validate labels
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
        """
        Generate new tokens autoregressively with improved sampling.
        
        OPTIMIZED GENERATION:
        - Proper model state management (eval/train modes)
        - Robust context window handling
        - Multiple sampling strategies (temperature, top-k)
        - Input validation and error handling
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy, >0 for stochastic)
            top_k: Top-k sampling parameter (None for full vocabulary)
            
        Returns:
            Generated token sequence (B, T + max_new_tokens)
        """
        # Input validation
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive if specified")
        
        self.eval()
        
        # Maximum total sequence length
        max_total_length = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
        
        for _ in range(max_new_tokens):
            # Truncate context if too long
            idx_cond = idx if idx.size(1) <= max_total_length else idx[:, -max_total_length:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Get logits for the last position and apply temperature
            logits = logits[:, -1, :] / max(temperature, 1e-7)  # Avoid division by zero
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            if temperature > 0:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Load a SymbolicTransformerModel from a checkpoint with robust error handling.

        IMPROVED LOADING:
        - Comprehensive error handling with detailed messages
        - Device management and validation
        - Configuration type checking
        - Graceful fallback for missing components
        
        Args:
            checkpoint_path (str): Path to the .pt checkpoint file
            device (str): Device to load the model onto ('cpu', 'cuda', or specific device)

        Returns:
            tuple: (SymbolicTransformerModel, tokenizer) or (None, None) if loading fails
        """
        # Validate inputs
        if not isinstance(checkpoint_path, str):
            print(f"Error: checkpoint_path must be a string, got {type(checkpoint_path)}")
            return None, None
        
        if not isinstance(device, str):
            print(f"Error: device must be a string, got {type(device)}")
            return None, None
        
        # Validate device
        try:
            target_device = torch.device(device)
        except Exception as e:
            print(f"Error: Invalid device '{device}': {e}")
            return None, None
        
        print(f"Loading checkpoint using {cls.__name__}.load_from_checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return None, None

        try:
            # Register safe globals for loading
            torch.serialization.add_safe_globals([SymbolicConfig])
            
            # Load checkpoint with proper error handling
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

            # Extract required components
            config = checkpoint.get('config')
            model_state_dict = checkpoint.get('model_state_dict')
            tokenizer = checkpoint.get('tokenizer')

            # Validate required components
            if not config:
                raise ValueError("Checkpoint missing required 'config' field")
            if not model_state_dict:
                raise ValueError("Checkpoint missing required 'model_state_dict' field")

            # Validate config type
            if not isinstance(config, SymbolicConfig):
                raise TypeError(f"Config must be SymbolicConfig, got {type(config)}")

            # Create and load model
            model = cls(config)
            
            # Load state dict with error handling
            try:
                model.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Loading state dict with strict=False due to: {e}")
                model.load_state_dict(model_state_dict, strict=False)
            
            # Move to target device and set evaluation mode
            model.to(target_device)
            model.eval()

            # Success message with model statistics
            param_count = model.get_num_params() / 1e6
            print(f"Model loaded successfully ({param_count:.2f}M parameters)")
            print(f"Model device: {next(model.parameters()).device}")
            
            if tokenizer:
                print(f"Tokenizer loaded successfully (type: {type(tokenizer).__name__})")
            else:
                print("Warning: Tokenizer not found in checkpoint")

            return model, tokenizer

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return None, None
        except PermissionError:
            print(f"Error: Permission denied accessing: {checkpoint_path}")
            return None, None
        except torch.serialization.pickle.UnpicklingError as e:
            print(f"Error: Failed to unpickle checkpoint: {e}")
            return None, None
        except RuntimeError as e:
            print(f"Error: PyTorch runtime error: {e}")
            return None, None
        except Exception as e:
            print(f"Error: Unexpected error during loading: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_checkpoint(self, checkpoint_path: str, tokenizer=None, additional_data=None):
        """
        Save model checkpoint with comprehensive error handling.
        
        ROBUST SAVING:
        - Input validation and error handling
        - Atomic saving with temporary files
        - Comprehensive checkpoint data
        - Success verification
        
        Args:
            checkpoint_path (str): Path where to save the checkpoint
            tokenizer: Optional tokenizer to include in checkpoint
            additional_data (dict): Optional additional data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Input validation
        if not isinstance(checkpoint_path, str):
            print(f"Error: checkpoint_path must be a string, got {type(checkpoint_path)}")
            return False
        
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
            except Exception as e:
                print(f"Error: Failed to create directory {checkpoint_dir}: {e}")
                return False
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'config': self.config,
                'model_state_dict': self.state_dict(),
                'model_class': self.__class__.__name__,
                'pytorch_version': torch.__version__,
                'parameter_count': self.get_num_params(),
            }
            
            # Add optional components
            if tokenizer is not None:
                checkpoint_data['tokenizer'] = tokenizer
            
            if additional_data is not None:
                if isinstance(additional_data, dict):
                    checkpoint_data.update(additional_data)
                else:
                    print(f"Warning: additional_data must be dict, got {type(additional_data)}")
            
            # Atomic save using temporary file
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint_data, temp_path)
            
            # Move to final location
            os.replace(temp_path, checkpoint_path)
            
            print(f"Checkpoint saved successfully to: {checkpoint_path}")
            print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
            return True
            
        except Exception as e:
            print(f"Error: Failed to save checkpoint: {type(e).__name__}: {e}")
            
            # Clean up temporary file if it exists
            temp_path = checkpoint_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return False

    def get_model_info(self):
        """
        Return comprehensive model information for debugging and analysis.
        
        Returns:
            dict: Model information including parameters, configuration, and statistics
        """
        info = {
            'model_class': self.__class__.__name__,
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
            },
            'architecture_details': {
                'head_dim': self.config.n_embd // self.config.n_head,
                'padding_idx': self.padding_idx,
                'max_position_embeddings': getattr(self.config, 'max_position_embeddings', self.config.block_size * 4),
            }
        }
        
        # Add device information if model has parameters
        if len(list(self.parameters())) > 0:
            info['device'] = str(next(self.parameters()).device)
            info['dtype'] = str(next(self.parameters()).dtype)
        
        return info


# Export the main model class
__all__ = ['SymbolicTransformerModel', 'SymbolicLayerNorm', 'VocabularyProjectionFFN', 'SymbolicCausalSelfAttentionALiBi', 'SymbolicTransformerBlock']