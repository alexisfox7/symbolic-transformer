#./config.py
"""
Configuration settings for Symbolic Transformer with ALiBi positional encoding.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch
import math

@dataclass
class TransformerConfig:
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
    #REVIEW is this used properly
    max_position_embeddings: Optional[int] = None  # Max sequence length (None = 4x block_size)
    
    # Symbolic-specific parameters
    use_v: Literal["none", "normal", "kronecker"] = "none"     # V matrix parameterization type
    use_proj: Literal["none", "normal", "kronecker"] = "none" # Output projection type
    
    # Attention parameters
    use_sparsemax: bool = False              # Use sparsemax instead of softmax in attention
    learnable_temperature: bool = False      # Use learnable temperature in attention

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
        
        valid_options = ["none", "normal", "kronecker"]
        if self.use_v not in valid_options:
            raise ValueError(f"use_v must be one of {valid_options}, got {self.use_v}")
        if self.use_proj not in valid_options:
            raise ValueError(f"use_proj must be one of {valid_options}, got {self.use_proj}")
    
    def update_from_tokenizer(self, tokenizer):
        """Update configuration from tokenizer."""
        if hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, '__len__'):
            self.vocab_size = len(tokenizer)

PRESETS = {
    'tiny': TransformerConfig(
        n_layer=2, n_head=2, n_embd=128, block_size=64,
        batch_size=64, learning_rate=5e-4, num_epochs=3
    ),
    'small': TransformerConfig(
        n_layer=6, n_head=6, n_embd=192, block_size=128,
        batch_size=16, learning_rate=3e-4, num_epochs=5
    ),
    'medium': TransformerConfig(
        n_layer=6, n_head=6, n_embd=384, block_size=128,
        batch_size=16, learning_rate=2e-4, num_epochs=8
    ),
    'large': TransformerConfig(
        n_layer=12, n_head=12, n_embd=768, block_size=256,
        batch_size=8, learning_rate=1e-4, num_epochs=10
    ),
}

def create_config_from_args(args) -> TransformerConfig:
    """Create config from command line arguments."""
    # start with preset (optional) so it can be overriden later if needed
    if hasattr(args, 'preset') and args.preset:
        config = get_preset_config(args.preset)
    else:
        config = TransformerConfig()
    
    # override with any provided arguments
    fields_to_copy = [
        # core 
        'n_embd', 'n_head', 'n_layer', 'block_size', 'dropout', 'bias',
        
        # symbolic flags
        'use_v', 'use_proj',
        
        # Training parameters
        'batch_size', 'num_epochs', 'learning_rate', 'weight_decay',
        
        # Generation parameters
        'temperature', 'top_k',
    ]

    for name in fields_to_copy:
        if hasattr(args, name) and getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    
    return config

def get_preset_config(preset_name: str) -> TransformerConfig:
    """Get predefined configuration preset."""
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESETS[preset_name]

def print_config(config: TransformerConfig, dataset_name: str = None, model=None):
    print("=" * 60)
    print("TRANSFORMER CONFIGURATION")
    print("=" * 60)
    
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    print(f"  Layers:              {config.n_layer}") 
    print(f"  Attention Heads:     {config.n_head}")
    print(f"  Embedding Dim:       {config.n_embd}")
    print(f"  Head Dimension:      {config.n_embd // config.n_head}")
    print(f"  Vocabulary Size:     {config.vocab_size or 'TBD'}")
    print(f"  Max Sequence:        {config.block_size}")
    print(f"  Dropout:             {config.dropout}")
    print(f"  Bias in Linear:      {config.bias}")
    
    print(f"\n🔬 SYMBOLIC FEATURES:")
    print(f"  Value Matrix:         {config.use_v}")
    print(f"  Projection Matrix:    {config.use_proj}")
    
    print(f"\n🏋️  TRAINING SETUP:")
    print(f"  Batch Size:          {config.batch_size}")
    print(f"  Epochs:              {config.num_epochs}")
    print(f"  Learning Rate:       {config.learning_rate}")
    print(f"  Weight Decay:        {config.weight_decay}")
    
    print(f"\n🎲 GENERATION:")
    print(f"  Temperature:         {config.temperature}")
    print(f"  Top-K:               {config.top_k}")
    
    if dataset_name:
        print(f"\n📊 DATASET:")
        print(f"  Name:                {dataset_name}")
    
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📏 MODEL SIZE:")
        print(f"  Parameters:          {total_params/1e6:.2f}M")
    
    print("=" * 60)

