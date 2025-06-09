# ./config.py
"""
Configuration settings for Symbolic Transformer with ALiBi positional encoding.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import math


@dataclass
class SymbolicConfig:
    """Configuration class for Symbolic Transformer models."""
    
    # Model architecture
    block_size: int = 128                    # Training sequence length
    vocab_size: Optional[int] = None         # Vocabulary size (set by tokenizer)
    n_layer: int = 6                         # Number of transformer layers
    n_head: int = 6                          # Number of attention heads
    n_embd: int = 384                        # Embedding dimension
    dropout: float = 0.1                     # Dropout probability
    bias: bool = False                       # Use bias in linear layers
    
    # ALiBi parameters
    max_position_embeddings: Optional[int] = None  # Max sequence length (None = 4x block_size)
    
    # Symbolic-specific parameters
    use_symbolic_ffn: bool = True            # Use vocabulary-constrained FFN
    use_vocab_refinement: bool = False       # Use refinement in projections
    use_v: bool = True                       # Use value projection constraints
    use_proj: bool = True                    # Use output projection constraints
    
    # Training parameters
    batch_size: int = 32                     # Batch size
    num_epochs: int = 5                      # Training epochs
    learning_rate: float = 3e-4              # Learning rate
    weight_decay: float = 0.01               # Weight decay
    
    # Generation parameters
    temperature: float = 0.8                 # Sampling temperature
    top_k: int = 50                          # Top-k sampling
    
    def __post_init__(self):
        """Post-initialization validation."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.block_size * 4
    
    def update_from_tokenizer(self, tokenizer):
        """Update configuration from tokenizer."""
        if hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, '__len__'):
            self.vocab_size = len(tokenizer)


def get_preset_config(preset_name: str) -> SymbolicConfig:
    """Get predefined configuration presets."""
    
    presets = {
        'tiny': SymbolicConfig(
            n_layer=2, n_head=2, n_embd=128,
            block_size=64, batch_size=64,
            learning_rate=5e-4, num_epochs=3
        ),
        'small': SymbolicConfig(
            n_layer=6, n_head=6, n_embd=192,
            block_size=128, batch_size=16,
            learning_rate=3e-4, num_epochs=5
        ),
        'medium': SymbolicConfig(
            n_layer=6, n_head=6, n_embd=384,
            block_size=128, batch_size=16,
            learning_rate=2e-4, num_epochs=8
        ),
        'large': SymbolicConfig(
            n_layer=12, n_head=12, n_embd=768,
            block_size=256, batch_size=8,
            learning_rate=1e-4, num_epochs=10
        ),
    }
    
    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return presets[preset_name]


def print_config(config: SymbolicConfig, dataset_name: str = None, model = None):
    """Print configuration in a formatted way."""
    print("=" * 60)
    print("SYMBOLIC TRANSFORMER CONFIGURATION")
    print("=" * 60)
    
    print(f"\n MODEL ARCHITECTURE:")
    print(f"  Layers:              {config.n_layer}")
    print(f"  Attention Heads:     {config.n_head}")
    print(f"  Embedding Dim:       {config.n_embd}")
    print(f"  Head Dimension:      {config.n_embd // config.n_head}")
    print(f"  Vocabulary Size:     {config.vocab_size}")
    print(f"  Block Size:          {config.block_size}")
    print(f"  Max Position:        {config.max_position_embeddings}")
    
    print(f"\n SYMBOLIC CONSTRAINTS:")
    print(f"  Symbolic FFN:        {config.use_symbolic_ffn}")
    print(f"  Vocab Refinement:    {config.use_vocab_refinement}")
    print(f"  Value Constraints:   {config.use_v}")
    print(f"  Output Constraints:  {config.use_proj}")
    print(f"  ALiBi Encoding:      Yes (no learned positions)")
    
    print(f"\n TRAINING:")
    print(f"  Batch Size:          {config.batch_size}")
    print(f"  Epochs:              {config.num_epochs}")
    print(f"  Learning Rate:       {config.learning_rate}")
    print(f"  Weight Decay:        {config.weight_decay}")
    print(f"  Dropout:             {config.dropout}")
    
    if dataset_name:
        print(f"\n DATASET:")
        print(f"  Dataset:             {dataset_name}")
    
    # Estimate parameters
    if config.vocab_size:
        params = estimate_parameters(config)
        print(f"\n MODEL SIZE:")
        print(f"  Estimated Params:    {params/1e6:.2f}M")
    
    if model is not None:
        print(f"\n MODEL SIZE:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Actual Params:       {total_params/1e6:.2f}M")
    
    print("=" * 60)


def estimate_parameters(config: SymbolicConfig) -> int:
    """Estimate number of model parameters."""
    vocab_size = config.vocab_size or 50257
    
    # Token embeddings (shared with lm_head)
    token_params = vocab_size * config.n_embd
    
    # Transformer layers
    layer_params = 0
    for _ in range(config.n_layer):
        # Layer norms (symbolic: n_head parameters each)
        ln_params = 2 * config.n_head * (2 if config.bias else 1)
        
        # Attention (Q, K projections only by default)
        attn_params = config.n_embd * 2 * config.n_embd + (2 * config.n_embd if config.bias else 0)
        
        # Symbolic projections if enabled
        if config.use_v:
            attn_params += config.n_head * config.n_head  # V matrix
        if config.use_proj:
            attn_params += config.n_head * config.n_head  # Output matrix
        
        # Symbolic FFN
        if config.use_symbolic_ffn:
            ffn_params = config.n_embd * config.n_embd + config.n_embd * vocab_size
            if config.bias:
                ffn_params += config.n_embd
        else:
            ffn_params = 0
        
        layer_params += ln_params + attn_params + ffn_params
    
    # Final layer norm
    final_ln = config.n_head * (2 if config.bias else 1)
    
    return token_params + layer_params + final_ln
