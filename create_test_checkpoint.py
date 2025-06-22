#!/usr/bin/env python3
"""
Create a test checkpoint for hook inference testing.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config.config import TransformerConfig
from src.mytokenizers import create_tokenizer


def create_test_checkpoint(model_type='vanilla', output_path='test_checkpoint.pt'):
    """Create a small test checkpoint."""
    
    # Create tokenizer first to get correct vocab size
    tokenizer = create_tokenizer('character')
    
    # Create config with correct vocab size
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    # Create model
    model = get_model(model_type, config)
    
    # Create checkpoint in the expected format
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,  # Save as dict for JSON serialization
        'training_result': {
            'model_type': model_type,
            'final_loss': 2.5,
            'best_val_loss': 2.3,
            'epochs_trained': 1
        },
        'epoch': 1,
        'step': 100
    }
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    print(f"Test checkpoint saved to: {output_path}")
    print(f"Model type: {model_type}")
    print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
    
    return output_path


if __name__ == "__main__":
    # Create both vanilla and symbolic test checkpoints
    vanilla_path = create_test_checkpoint('vanilla', 'test_vanilla_checkpoint.pt')
    symbolic_path = create_test_checkpoint('symbolic', 'test_symbolic_checkpoint.pt')
    
    print(f"\nCreated test checkpoints:")
    print(f"  Vanilla: {vanilla_path}")
    print(f"  Symbolic: {symbolic_path}")
    print(f"\nYou can now test with:")
    print(f"  python run_inference_with_hooks.py {vanilla_path} --prompt 'Hello world'")
    print(f"  python run_inference_with_hooks.py {symbolic_path} --prompt 'Hello world'")