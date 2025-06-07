#!/usr/bin/env python
# ./demo_symbolic.py
"""
Simple demonstration of the Symbolic Transformer.
"""

import torch
from config import get_preset_config, print_config
from model import get_model


def main():
    """Run a simple demo of the Symbolic Transformer."""
    
    print("=" * 50)
    print("SYMBOLIC TRANSFORMER DEMO")
    print("=" * 50)
    
    # Create a tiny configuration for demo
    config = get_preset_config('tiny')
    config.vocab_size = 1000  # Small vocab for demo
    
    print_config(config, dataset_name="Demo")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating Symbolic Transformer...")
    model = get_model("Symbolic", config=config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print("✓ Forward pass successful!")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5), device=device)
    
    generated = model.generate(
        prompt,
        max_new_tokens=8,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"Length: {prompt.size(1)} → {generated.size(1)}")
    print(" Generation successful!")
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    
    # Create labels for training
    labels = input_ids.clone()
    
    # Forward pass with loss
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Backward pass
    loss.backward()
    print(" Backward pass successful!")
    
    # Test symbolic components
    print("\nTesting symbolic components...")
    
    with torch.no_grad():
        # Test vocabulary projection
        test_input = torch.randn(1, 5, config.n_embd, device=device)
        vocab_output = model.vocab_grounding(test_input)
        
        print(f"Vocab projection: {test_input.shape} → {vocab_output.shape}")
        
        # Test symbolic layer norm
        ln_output = model.transformer.ln_f(test_input)
        print(f"Symbolic LayerNorm: {test_input.shape} → {ln_output.shape}")
        
        print("✓ Symbolic components working!")
    
    print("\n" + "=" * 50)
    print(" ALL TESTS PASSED!")
    print("The Symbolic Transformer is working correctly.")
    print("\nNext steps:")
    print("1. Run: python train_symbolic.py --preset tiny --max_samples 1000")
    print("2. Or:  python train_symbolic.py --preset small")
    print("=" * 50)


if __name__ == "__main__":
    main()
