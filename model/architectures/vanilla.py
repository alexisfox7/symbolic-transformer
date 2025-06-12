import torch
import torch.nn as nn
from torch.nn import functional as F

from ..base import TransformerBase
from ..components import VanillaAttention, VanillaNorm, VanillaFFN

class VanillaTransformerBlock(nn.Module):
    """Standard transformer block with pre-layer normalization."""
    #TODO: check functionality with base
    def __init__(self, config):
        super().__init__()
        self.ln_1 = VanillaNorm(config.n_embd, bias=config.bias)
        self.attn = VanillaAttention(config)
        self.ln_2 = VanillaNorm(config.n_embd, bias=config.bias)
        self.ffn = VanillaFFN(config)

    def forward(self, x):
        # Using pre-layer norm
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class VanillaTransformerModel(TransformerBase):
    """
    Vanilla Transformer model for baseline comparison.
    Has:
    - Token and positional embeddings
    - Multi-head causal self-attention
    - Feed-forward networks
    - Pre-layer normalization
    """

    #TODO: old code
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)
        
        # Core model components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([VanillaTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=VanillaNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to output projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"VanillaTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Architecture: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")


    #TODO: old code
    #REVIEW why is there attention mask, also why was there previous issue clayton mentioned
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the VanillaTransformerModel.
        
        Args:
            input_ids: Token input IDs (B, T)
            attention_mask: Attention mask (currently unused)
            labels: Target labels for language modeling loss (B, T)
            
        Returns:
            Dictionary containing loss and logits
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Check sequence length limits
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")

        # Token embeddings
        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        
        # Positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        
        # Combine embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

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

   