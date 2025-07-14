import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from .base import TransformerBase
from ..components import VanillaAttention, VanillaNorm, VanillaFFN

class VanillaTransformerBlock(nn.Module):
    """Standard transformer block with pre-layer normalization."""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = VanillaNorm(config.n_embd, bias=config.bias)
        self.attn = VanillaAttention(config)
        self.ln_2 = VanillaNorm(config.n_embd, bias=config.bias)
        self.ffn = VanillaFFN(config)

    def forward(self, x, layer_idx=None, hook_manager=None, hook_state=None):
        # using pre-layer norm
        x_norm = self.ln_1(x)
        attn_out = self.attn(x_norm, layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        x = x + attn_out
        
        x_norm = self.ln_2(x)
        ffn_out = self.ffn(x_norm, layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        x = x + ffn_out
        return x


class VanillaTransformer(TransformerBase):
    """
    Vanilla Transformer model for baseline comparison.
    Has:
    - Token and positional embeddings
    - Multi-head causal self-attention
    - Feed-forward networks
    - Pre-layer norm
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd), # position embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([VanillaTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=VanillaNorm(config.n_embd, bias=config.bias),
        ))
        
        # language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # init weights
        self.apply(self._init_weights)
        self._apply_projection_init()

        print(f"VanillaTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Architecture: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")

    def forward(self, input_ids, targets=None, hook_state=None):
        device = input_ids.device
        b, t = input_ids.size()

        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb)

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, layer_idx=layer_idx, hook_manager=self.hook_manager, hook_state=hook_state)

            # for logit lens
            if self.hook_manager:
                for hook in self.hook_manager.hooks:
                    if hasattr(hook, 'analyze_layer') and hook.enabled:
                        tokens = hook_state.get('tokens', []) if hook_state else []
                        position = hook_state.get('position', 0) if hook_state else 0
                        hook.analyze_layer(x, layer_idx, position, tokens)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # calculate lm loss 
        loss = None
        if targets is not None:
            if targets[0, 0] != input_ids[0, 0]:
                raise ValueError("Targets may be pre-shifted. Expected targets[0,0] == input_ids[0,0]")
            if targets.shape != input_ids.shape:
                raise ValueError(f"Targets shape {targets.shape} must match input_ids shape {input_ids.shape}")
            
            # shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

   