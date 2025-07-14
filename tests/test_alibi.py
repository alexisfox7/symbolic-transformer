#!/usr/bin/env python3
"""
Debug ALiBi slopes and attention parameters from a checkpoint.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def debug_alibi_from_checkpoint(checkpoint_path):
    """Extract and analyze ALiBi slopes from checkpoint."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Find ALiBi slopes in the state dict
    alibi_slopes = {}
    attention_params = {}
    
    for key, value in state_dict.items():
        if 'alibi_slopes' in key:
            layer_info = key.split('.')
            layer_num = None
            for part in layer_info:
                if part.startswith('h') and part[1:].isdigit():
                    layer_num = int(part[1:])
                    break
            
            if layer_num is not None:
                alibi_slopes[layer_num] = value.clone()
                print(f"Layer {layer_num} ALiBi slopes: {value}")
        
        # Also collect other attention parameters
        if 'attn' in key and any(param in key for param in ['v_tmp', 'proj_tmp', 'c_attn', 'c_proj']):
            attention_params[key] = value.clone()
    
    # Analyze ALiBi slopes
    if alibi_slopes:
        print(f"\n=== ALiBi Analysis ===")
        print(f"Found ALiBi slopes for {len(alibi_slopes)} layers")
        
        # Check if slopes are reasonable
        all_slopes = []
        for layer, slopes in alibi_slopes.items():
            print(f"\nLayer {layer}:")
            print(f"  Slopes: {slopes}")
            print(f"  Range: [{slopes.min().item():.6f}, {slopes.max().item():.6f}]")
            print(f"  Mean: {slopes.mean().item():.6f}")
            print(f"  Std: {slopes.std().item():.6f}")
            all_slopes.extend(slopes.tolist())
        
        # Plot ALiBi slopes across layers and heads
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: All slopes distribution
        axes[0, 0].hist(all_slopes, bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of All ALiBi Slopes')
        axes[0, 0].set_xlabel('Slope Value')
        axes[0, 0].set_ylabel('Count')
        
        # Plot 2: Slopes by layer
        layers = sorted(alibi_slopes.keys())
        for layer in layers:
            slopes = alibi_slopes[layer]
            axes[0, 1].plot(range(len(slopes)), slopes, 'o-', label=f'Layer {layer}')
        axes[0, 1].set_title('ALiBi Slopes by Head (Each Layer)')
        axes[0, 1].set_xlabel('Head Index')
        axes[0, 1].set_ylabel('Slope Value')
        axes[0, 1].legend()
        
        # Plot 3: Layer-wise slope statistics
        layer_means = [alibi_slopes[l].mean().item() for l in layers]
        layer_stds = [alibi_slopes[l].std().item() for l in layers]
        
        axes[1, 0].plot(layers, layer_means, 'bo-', label='Mean')
        axes[1, 0].fill_between(layers, 
                               [m-s for m,s in zip(layer_means, layer_stds)],
                               [m+s for m,s in zip(layer_means, layer_stds)], 
                               alpha=0.3)
        axes[1, 0].set_title('ALiBi Slope Statistics by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Slope Value')
        axes[1, 0].legend()
        
        # Plot 4: Check for expected ALiBi pattern
        # ALiBi should follow pattern: 2^(-8*i/n_heads) where i is head index
        if layers:
            first_layer_slopes = alibi_slopes[layers[0]]
            n_heads = len(first_layer_slopes)
            expected_slopes = []
            for i in range(n_heads):
                expected = 2**(-8*i/n_heads)
                expected_slopes.append(expected)
            
            axes[1, 1].plot(range(n_heads), first_layer_slopes.numpy(), 'ro-', label='Actual (Layer 0)')
            axes[1, 1].plot(range(n_heads), expected_slopes, 'b--', label='Expected ALiBi Pattern')
            axes[1, 1].set_title('ALiBi Pattern Check (First Layer)')
            axes[1, 1].set_xlabel('Head Index')
            axes[1, 1].set_ylabel('Slope Value')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('alibi_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("❌ No ALiBi slopes found in checkpoint!")
        print("Available keys with 'alibi' or 'attn':")
        for key in state_dict.keys():
            if 'alibi' in key.lower() or 'attn' in key.lower():
                print(f"  {key}")
    
    # Check other attention parameters
    print(f"\n=== Attention Parameters ===")
    if attention_params:
        for key, value in attention_params.items():
            print(f"{key}: shape={value.shape}, norm={value.norm().item():.4f}")
            if 'v_tmp' in key or 'proj_tmp' in key:
                # These are the Kronecker parameters
                print(f"  Kronecker matrix condition number: {torch.linalg.cond(value.float()).item():.2f}")
                print(f"  Eigenvalues: {torch.linalg.eigvals(value.float()).real}")
    
    return alibi_slopes, attention_params

def simulate_alibi_bias(slopes, seq_len=10):
    """Simulate what ALiBi bias looks like with given slopes."""
    print(f"\n=== ALiBi Bias Simulation (seq_len={seq_len}) ===")
    
    # Create position matrix
    positions = torch.arange(seq_len).float()
    relative_pos = positions[None, :] - positions[:, None]  # [seq_len, seq_len]
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Show bias for each head
    for head_idx, slope in enumerate(slopes[:6]):  # Show first 6 heads
        bias = slope * relative_pos
        bias = bias.masked_fill(causal_mask == 0, float('-inf'))
        
        im = axes[head_idx].imshow(bias.numpy(), cmap='RdBu', aspect='auto')
        axes[head_idx].set_title(f'Head {head_idx}, Slope: {slope:.4f}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[head_idx])
    
    plt.tight_layout()
    plt.savefig('alibi_bias_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_alibi_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    try:
        alibi_slopes, attention_params = debug_alibi_from_checkpoint(checkpoint_path)
        
        # Simulate bias for first layer if available
        if alibi_slopes:
            first_layer = min(alibi_slopes.keys())
            simulate_alibi_bias(alibi_slopes[first_layer])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()