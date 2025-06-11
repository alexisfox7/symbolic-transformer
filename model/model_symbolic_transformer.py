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


class VocabularyProjectionFFN(nn.Module):
    """
    Feed Forward Network (FFN) that constrains outputs to the vocabulary embedding manifold
    with proper head-channel decomposition. Each head channel operates independently on
    its corresponding slice of the vocabulary embeddings.
    """
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)

        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        # Store reference to vocabulary embeddings (not a copy)
        self.vocab_embeddings_ref = vocab_embeddings_ref

        # Per-channel FFN transformations
        self.channel_ffns = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
            for _ in range(self.n_head)
        ])

        # Per-channel vocabulary attention projections
        self.channel_vocab_attentions = nn.ModuleList([
            nn.Linear(self.head_dim, self.vocab_size, bias=False)
            for _ in range(self.n_head)
        ])

        # Per-channel learnable temperature for attention sharpness
        self.channel_temperatures = nn.Parameter(torch.ones(self.n_head))

        # Optional refinement layers per channel
        self.use_refinement = getattr(config, 'use_vocab_refinement', False)
        if self.use_refinement:
            self.channel_refinements = nn.ModuleList([
                nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
                for _ in range(self.n_head)
            ])
            self.channel_refinement_gates = nn.ModuleList([
                nn.Linear(self.head_dim, self.head_dim, bias=config.bias)
                for _ in range(self.n_head)
            ])

    def _get_vocab_channel(self, channel_idx):
        """
        Extract the channel_idx-th slice of vocabulary embeddings.

        Args:
            channel_idx: Index of the head channel (0 to n_head-1)

        Returns:
            E_h: Vocabulary embeddings for channel h, shape (vocab_size, head_dim)
        """
        start_idx = channel_idx * self.head_dim
        end_idx = (channel_idx + 1) * self.head_dim
        return self.vocab_embeddings_ref.weight[:, start_idx:end_idx]  # (vocab_size, head_dim)

    def forward(self, x):
        """
        Project input to vocabulary embedding manifold via channel-wise learned attention.

        Args:
            x: Input tensor (B, T, n_embd)

        Returns:
            Vocabulary-grounded output tensor (B, T, n_embd)
        """
        B, T, C = x.shape
        assert C == self.n_embd, f"Input dim {C} != expected {self.n_embd}"

        # Reshape to separate head channels: (B, T, n_head, head_dim)
        x_channels = x.view(B, T, self.n_head, self.head_dim)

        channel_outputs = []

        for h in range(self.n_head):
            # Extract channel h: (B, T, head_dim)
            x_h = x_channels[:, :, h, :]

            # Apply channel-specific FFN transformation
            ffn_h = self.channel_ffns[h](x_h)  # (B, T, head_dim)

            # Get vocabulary embeddings for this channel
            E_h = self._get_vocab_channel(h)  # (vocab_size, head_dim)

            # Compute attention weights over vocabulary for this channel
            vocab_logits_h = self.channel_vocab_attentions[h](ffn_h)  # (B, T, vocab_size)

            # Apply temperature scaling
            temp_h = torch.clamp(self.channel_temperatures[h], min=0.1)
            vocab_weights_h = F.softmax(vocab_logits_h / temp_h, dim=-1)  # (B, T, vocab_size)

            # Project to vocabulary manifold for this channel
            vocab_output_h = torch.matmul(vocab_weights_h, E_h)  # (B, T, head_dim)

            # Optional refinement while maintaining vocabulary grounding
            if self.use_refinement:
                refinement_h = torch.tanh(self.channel_refinements[h](x_h))
                gate_h = torch.sigmoid(self.channel_refinement_gates[h](x_h))
                vocab_output_h = vocab_output_h * (1 - gate_h) + refinement_h * gate_h

                # Re-project to ensure vocabulary grounding is maintained
                refined_logits_h = self.channel_vocab_attentions[h](vocab_output_h)
                refined_weights_h = F.softmax(refined_logits_h / temp_h, dim=-1)
                vocab_output_h = torch.matmul(refined_weights_h, E_h)

            channel_outputs.append(vocab_output_h)

        # Concatenate all channel outputs: (B, T, n_head, head_dim) -> (B, T, n_embd)
        output = torch.stack(channel_outputs, dim=2).view(B, T, self.n_embd)

        return self.dropout(output)

#class VocabularyProjectionFFN(nn.Module):
#    """
#    Feed Forward Network (FFN) that constrains outputs to the vocabulary embedding manifold.
#    This ensures all FFN outputs remain symbolically interpretable.
#    """
#    def __init__(self, config, vocab_embeddings_ref):
#        super().__init__()
#        self.vocab_size = config.vocab_size
#        self.n_embd = config.n_embd
#        self.dropout = nn.Dropout(config.dropout)
#        
#        # Store reference to vocabulary embeddings (not a copy)
#        self.vocab_embeddings_ref = vocab_embeddings_ref
#        
#        # Intermediate projection for computing vocabulary attention
#        self.vocab_query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#        self.vocab_attention = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#        
#        # Learnable temperature for attention sharpness
#        self.temperature = nn.Parameter(torch.ones(1))
#        
#        # Optional refinement layers
#        self.use_refinement = getattr(config, 'use_vocab_refinement', False)
#        if self.use_refinement:
#            self.refinement = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#            self.refinement_gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#    
#    def forward(self, x):
#        """
#        Project input to vocabulary embedding manifold via learned attention.
#        
#        Args:
#            x: Input tensor (B, T, n_embd)
#            
#        Returns:
#            Vocabulary-grounded output tensor (B, T, n_embd)
#        """
#        # Compute query for vocabulary attention
#        query = self.vocab_query(x)  # (B, T, n_embd)
#        
#        # Compute attention weights over vocabulary
#        vocab_logits = self.vocab_attention(query)  # (B, T, vocab_size)
#        vocab_weights = F.softmax(vocab_logits / torch.clamp(self.temperature, min=0.1), dim=-1)
#        
#        # Project to vocabulary manifold
#        vocab_output = torch.matmul(vocab_weights, self.vocab_embeddings_ref.weight)  # (B, T, n_embd)
#        
#        # Optional refinement while maintaining vocabulary grounding
#        if self.use_refinement:
#            refinement = torch.tanh(self.refinement(x))
#            gate = torch.sigmoid(self.refinement_gate(x))
#            vocab_output = vocab_output * (1 - gate) + refinement * gate
#            
#            # Re-project to ensure vocabulary grounding is maintained
#            refined_logits = self.vocab_attention(vocab_output)
#            refined_weights = F.softmax(refined_logits / torch.clamp(self.temperature, min=0.1), dim=-1)
#            vocab_output = torch.matmul(refined_weights, self.vocab_embeddings_ref.weight)
#        
#        return self.dropout(vocab_output)


class SymbolicCausalSelfAttentionALiBi(nn.Module):
    """
    Causal self-attention mechanism for the Symbolic Transformer with ALiBi positional encoding.
    Operates on symbolic representations without vocabulary projection - attention is a 
    symbolic operation that routes and combines existing symbolic information.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

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
        Forward pass for symbolic attention.
        
        Args:
            x: Input symbolic state (B, T, n_embd)
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

        Args:
            xt: Symbolic input state (B, T, n_embd)

        Returns:
            Updated symbolic state (B, T, n_embd)
        """
        # Symbolic attention path
        norm_for_attn = self.ln_1(xt)
        attn_output = self.attn(norm_for_attn)  # Fixed: single argument
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
            h=nn.ModuleList([SymbolicTransformerBlock(config, None) for _ in range(config.n_layer)]),
            ln_f=SymbolicLayerNorm(config.n_embd, config.n_head, bias=config.bias),
        ))
        
        # Pass vocabulary embedding reference to all blocks after creation
        for block in self.transformer.h:
            block.attn.vocab_embeddings_ref = self.transformer.wte
            if hasattr(block.attn, 'symbolic_v_projection'):
                block.attn.symbolic_v_projection.vocab_embeddings_ref = self.transformer.wte
            if hasattr(block.attn, 'symbolic_output_projection'):
                block.attn.symbolic_output_projection.vocab_embeddings_ref = self.transformer.wte
            if hasattr(block, 'ffn'):
                block.ffn.vocab_embeddings_ref = self.transformer.wte

        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight 
        
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

        print(f"SymbolicTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Symbolic constraints: symbolic_ffn={getattr(config, 'use_symbolic_ffn', True)}")
        print(f"Vocabulary size: {config.vocab_size}, Embedding dim: {config.n_embd}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
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
            # Initialize symbolic attention parameters
            if hasattr(module, 'v_tmp'):
                torch.nn.init.normal_(module.v_tmp, mean=0.0, std=0.02)
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
        
        Args:
            input_ids: Token input IDs (B, T)
            attention_mask: Attention mask (unused with ALiBi causal attention)
            labels: Target labels for language modeling loss
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Check sequence length limits
        max_len = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
        if t > max_len:
            raise ValueError(f"Sequence length {t} exceeds maximum supported length {max_len}")

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
        Generate new tokens autoregressively.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """
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
            logits = logits[:, -1, :] / temperature
            
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
        Load a SymbolicTransformerModel from a checkpoint.

        Args:
            checkpoint_path (str): Path to the .pt checkpoint file
            device (str): Device to load the model onto ('cpu' or 'cuda')

        Returns:
            tuple: (SymbolicTransformerModel, tokenizer) or (None, None) if loading fails
        """
        target_device = torch.device(device)
        print(f"Loading checkpoint using {cls.__name__}.load_from_checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return None, None

        try:
            torch.serialization.add_safe_globals([SymbolicConfig])
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

            config = checkpoint.get('config')
            model_state_dict = checkpoint.get('model_state_dict')
            tokenizer = checkpoint.get('tokenizer')

            if not config or not model_state_dict:
                raise ValueError("Checkpoint must contain 'config' and 'model_state_dict'.")

            if not isinstance(config, SymbolicConfig):
                 raise TypeError(f"Config in checkpoint is not SymbolicConfig. Type: {type(config)}")

            model = cls(config)
            model.load_state_dict(model_state_dict)
            model.to(target_device)
            model.eval()

            print(f"Model loaded successfully via class method ({model.get_num_params()/1e6:.2f}M params).")
            if tokenizer:
                print(f"Tokenizer loaded successfully (type: {type(tokenizer)}).")
            else:
                print("Warning: Tokenizer not found in checkpoint.")

            return model, tokenizer

        except Exception as e:
            print(f"An unexpected error occurred during {cls.__name__}.load_from_checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# Export the main model class
__all__ = ['SymbolicTransformerModel']
