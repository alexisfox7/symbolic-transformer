import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..base import TransformerBase
from ..components import ChannelNorm, SymbolicAttention, VocabFFN

class SymbolicTransformerBlock(nn.Module):
    """
    Pure symbolic transformer block with vocabulary-constrained operations.
    Maintains single symbolic stream with FFN vocabulary projection.
    """
    #TODO: old code
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.config = config

        # Symbolic layer normalization
        self.ln_1 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)
        self.ln_2 = ChannelNorm(config.n_embd, config.n_head, bias=config.bias)

        # Symbolic attention mechanism (no vocab reference needed)
        self.attn = SymbolicAttention(config)

        # Optional vocabulary-constrained FFN
        self.use_symbolic_ffn = getattr(config, 'use_symbolic_ffn', True)
        if self.use_symbolic_ffn:
            self.ffn = VocabularyProjectionFFN(config, vocab_embeddings_ref)

    #TODO: old code
    def forward(self, xt):
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
    #TODO: old code
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
            ln_f=ChannelNorm(config.n_embd, config.n_head, bias=config.bias),
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

    #REVIEW does this need to be overridden?
    #TODO: old code
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
        elif isinstance(module, SymbolicAttention):
            # Initialize symbolic attention parameters
            if hasattr(module, 'v_tmp'):
                torch.nn.init.normal_(module.v_tmp, mean=0.0, std=0.02)
            if hasattr(module, 'proj_tmp'):
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, ChannelNorm):
            # Initialize symbolic layer norm
            torch.nn.init.ones_(module.channel_weights)
            if module.channel_biases is not None:
                torch.nn.init.zeros_(module.channel_biases)

    #TODO: old code
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
