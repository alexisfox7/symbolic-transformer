import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import TransformerBase
from ..components import ChannelNorm, TFTAttention, VanillaFFN

class TFTTransformerBlock(nn.Module):
    """
    Symbolic transformer block, FFN output is vocabulary-constrained.
    Contains a symbolic stream and embedding stream. 
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_2 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_3 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)

        self.attn = TFTAttention(config)

        self.ffn = VanillaFFN(config)

    def forward(self, xt, xe, layer_idx=None, hook_manager=None, hook_state=None):
        x_comb = self.ln_1(xt + xe)
        xt = self.ln_2(xt)
        
        # attention
        attn_out = self.attn(x_comb, xt, layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        xt = xt + attn_out
       
        xe = xt + xe
        xe = self.ln_3(xe)
        
        # ffn
        ffn_out = self.ffn(xe, layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        xe = ffn_out   

        return xt, xe

class TFTTransformer(TransformerBase):
    """
    Symbolic Transformer model.
    """
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte=self.wte,
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TFTTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=ChannelNorm(config.n_embd, config.n_head, bias=config.bias),
        ))

        # language model head (weight tying)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, config.bias)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self._apply_projection_init()
        
        #REVIEW if needed init temperature
        print(f"SymbolicTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Vocabulary size: {config.vocab_size}, Embedding dim: {config.n_embd}")

    #REVIEW does this need to be overridden?
    def _init_weights(self, module):
        """Initialize model weights with symbolic-aware initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, TFTAttention):
            if hasattr(module, 'v_tmp'):
                torch.nn.init.normal_(module.v_tmp, mean=0.0, std=0.02)
            if hasattr(module, 'proj_tmp'):
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02)
        elif isinstance(module, ChannelNorm):
            torch.nn.init.ones_(module.channel_weights)
            if module.channel_biases is not None:
                torch.nn.init.zeros_(module.channel_biases)

    def forward(self, input_ids, targets=None, hook_state=None):
        """
        Forward pass for the SymbolicTransformer.
        """
        device = input_ids.device
        b, t = input_ids.size()

        tok_emb = self.transformer.wte(input_ids)
        
        xt = self.transformer.drop(tok_emb)
        xe = torch.zeros_like(xt)

        for layer_idx, block in enumerate(self.transformer.h):
            xt, xe = block(xt, layer_idx=layer_idx, hook_manager=self.hook_manager, hook_state=hook_state)

        x_final = xt + xe
        x_final = self.transformer.ln_f(x_final)
        logits = self.lm_head(x_final)

        # lm loss
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
