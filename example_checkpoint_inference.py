#!/usr/bin/env python3
"""
Simple example of running inference with hooks on a checkpoint.
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import create_attention_extraction_hook
from src.mytokenizers import create_tokenizer


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load a model checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and create model
    config = TransformerConfig(**checkpoint['config'])
    model_type = checkpoint.get('model_type', 'vanilla')
    model = get_model(model_type, config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def main():
    # Path to your checkpoint
    checkpoint_path = "test_vanilla_checkpoint.pt"  # Using test checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Please update the checkpoint_path variable with your actual checkpoint path")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded: {model.get_num_params()/1e6:.2f}M parameters")
    
    # Create tokenizer (adjust based on what you used during training)
    tokenizer = create_tokenizer('character')  # or 'gpt2', or load from path
    
    # Create attention extraction hook
    attention_hook = create_attention_extraction_hook(
        threshold=0.1,  # Only track attention weights > 0.1
        store_values=False  # Set True to store value vectors
    )
    
    # Generate text with hooks
    prompt = "Hello world"
    print(f"\nGenerating from prompt: '{prompt}'")
    
    ids, text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        hooks=[attention_hook]  # Add your hooks here
    )
    
    print(f"\nGenerated text:\n{text}")
    
    # Analyze attention patterns
    print(f"\n--- Attention Analysis ---")
    print(f"Total attention records: {len(attention_hook.attention_data)}")
    
    # Example: Find which tokens "Hello" attended to most
    hello_summary = attention_hook.get_token_attention_summary("H")
    print(f"\nAttention summary for 'H':")
    print(f"  Total attention received: {hello_summary['total_received']:.4f}")
    print(f"  Total attention given: {hello_summary['total_given']:.4f}")
    
    # Example: Get attention patterns from layer 0, head 0
    layer0_head0_edges = attention_hook.get_edges_for_layer_head(0, 0)
    if layer0_head0_edges:
        print(f"\nLayer 0, Head 0 had {len(layer0_head0_edges)} attention edges")
        # Show top 5 strongest connections
        top_edges = sorted(layer0_head0_edges, key=lambda x: x['weight'], reverse=True)[:5]
        print("Top 5 attention connections:")
        for edge in top_edges:
            print(f"  '{edge['source_token']}' -> '{edge['target_token']}': {edge['weight']:.4f}")


if __name__ == "__main__":
    main()