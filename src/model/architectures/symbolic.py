import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .base import TransformerBase
from ..components import ChannelNorm, SymbolicAttention, VocabFFN

class SymbolicTransformerBlock(nn.Module):
    """
    Symbolic transformer block.
    FFN output is vocabulary-constrained (VocabFFN)
    Maintains single symbolic stream.
    """
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.config = config

        self.ln_1 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_2 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)

        self.attn = SymbolicAttention(config)
        self.ffn = VocabFFN(config, vocab_embeddings_ref)

    #REVIEW check this is right order
    def forward(self, x): # , layer_idx=None, hook_manager=None, hook_state=None):
        x_norm = self.ln_1(x)
        attn_out = self.attn(x_norm) # , layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        x = x + attn_out
        
        x_norm = self.ln_2(x)
        ffn_out = self.ffn(x_norm) # , layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
        x = x + ffn_out
        return x

class SymbolicTransformer(TransformerBase):
    """
    Symbolic Transformer model.
    """
    #TODO:old code
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte=self.wte,
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([SymbolicTransformerBlock(config, self.wte) for _ in range(config.n_layer)]),
            ln_f=ChannelNorm(config.n_embd, config.n_head, bias=config.bias),
        ))

        # language model head (weight tying)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, config.bias)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        self._apply_projection_init()
        
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
        elif isinstance(module, SymbolicAttention):
            if hasattr(module, 'v_tmp'):
                torch.nn.init.normal_(module.v_tmp, mean=0.0, std=0.02)
            if hasattr(module, 'proj_tmp'):
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02)
        elif isinstance(module, ChannelNorm):
            torch.nn.init.ones_(module.channel_weights)
            if module.channel_biases is not None:
                torch.nn.init.zeros_(module.channel_biases)

    def forward(self, input_ids, targets=None): # , hook_state=None):
        """
        Forward pass for the SymbolicTransformer.
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Hook: on_forward_begin
        state = {'targets': targets, 'model': self}
        self.hook_manager.call_hooks('on_forward_begin', input_ids, state)

        tok_emb = self.transformer.wte(input_ids)
        
        xt = self.transformer.drop(tok_emb)

        for layer_idx, block in enumerate(self.transformer.h):
            # Hook: on_layer_begin
            self.hook_manager.call_hooks('on_layer_begin', layer_idx, xt, state)
            
            xt = block(xt) # , layer_idx=layer_idx, hook_manager=self.hook_manager, hook_state=hook_state)
            
            # Hook: on_layer_end
            self.hook_manager.call_hooks('on_layer_end', layer_idx, xt, state)
        
        x_final = self.transformer.ln_f(xt)
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

        outputs = {'loss': loss, 'logits': logits}
        
        # Hook: on_forward_end
        self.hook_manager.call_hooks('on_forward_end', outputs, state)
        
        return outputs
