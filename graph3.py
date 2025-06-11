#!/usr/bin/env python3
"""
Script to analyze whether V*W_O ≈ Identity in trained vanilla transformer checkpoints.

This script loads a vanilla transformer checkpoint and analyzes the composition
of V (value projection) and W_O (output projection) matrices to see if their
product is close to the identity matrix, as suggested by recent research.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def extract_v_wo_matrices(model_state_dict, layer_idx, n_head, head_dim):
    """
    Extract V and W_O matrices for a specific layer from the checkpoint.
    
    In the vanilla transformer:
    - c_attn contains concatenated Q, K, V projections
    - c_proj is the output projection W_O
    """
    # Get the attention weights for this layer
    c_attn_weight = model_state_dict[f'transformer.h.{layer_idx}.attn.c_attn.weight']
    c_proj_weight = model_state_dict[f'transformer.h.{layer_idx}.attn.c_proj.weight']
    
    n_embd = c_attn_weight.shape[1]
    
    # Extract V from the concatenated QKV matrix
    # c_attn shape: (3*n_embd, n_embd) - projects to [Q, K, V]
    # We want the V part: indices [2*n_embd:3*n_embd, :]
    W_V = c_attn_weight[2*n_embd:3*n_embd, :].T  # Shape: (n_embd, n_embd)
    
    # W_O is the output projection
    W_O = c_proj_weight.T  # Shape: (n_embd, n_embd)
    
    return W_V, W_O

def analyze_identity_similarity(V_W_O, layer_idx):
    """
    Analyze how close V*W_O is to the identity matrix.
    """
    n_embd = V_W_O.shape[0]
    identity = torch.eye(n_embd, device=V_W_O.device, dtype=V_W_O.dtype)
    
    # Compute various similarity metrics
    results = {}
    
    # 1. Frobenius norm distance to identity
    diff = V_W_O - identity
    frobenius_dist = torch.norm(diff, p='fro').item()
    results['frobenius_distance'] = frobenius_dist
    
    # 2. Normalized Frobenius distance
    frobenius_norm_V_W_O = torch.norm(V_W_O, p='fro').item()
    frobenius_norm_identity = torch.norm(identity, p='fro').item()
    results['normalized_frobenius'] = frobenius_dist / frobenius_norm_identity
    
    # 3. Diagonal dominance - how much larger are diagonal elements
    diagonal_values = torch.diag(V_W_O)
    off_diagonal_mask = ~torch.eye(n_embd, dtype=bool)
    off_diagonal_values = V_W_O[off_diagonal_mask]
    
    results['mean_diagonal'] = diagonal_values.mean().item()
    results['std_diagonal'] = diagonal_values.std().item()
    results['mean_off_diagonal'] = off_diagonal_values.mean().item()
    results['std_off_diagonal'] = off_diagonal_values.std().item()
    
    # 4. Ratio of diagonal to off-diagonal magnitudes
    diagonal_magnitude = torch.abs(diagonal_values).mean().item()
    off_diagonal_magnitude = torch.abs(off_diagonal_values).mean().item()
    results['diagonal_to_off_diagonal_ratio'] = diagonal_magnitude / (off_diagonal_magnitude + 1e-8)
    
    # 5. Spectral properties
    eigenvalues = torch.linalg.eigvals(V_W_O)
    eigenvalues_real = eigenvalues.real
    results['largest_eigenvalue'] = eigenvalues_real.max().item()
    results['smallest_eigenvalue'] = eigenvalues_real.min().item()
    results['eigenvalue_mean'] = eigenvalues_real.mean().item()
    results['eigenvalue_std'] = eigenvalues_real.std().item()
    
    # 6. How many eigenvalues are close to 1 (identity has all eigenvalues = 1)
    close_to_one = torch.abs(eigenvalues_real - 1.0) < 0.1
    results['eigenvalues_close_to_one'] = close_to_one.sum().item() / len(eigenvalues_real)
    
    return results

def visualize_matrix(matrix, title, save_path=None):
    """Create visualization of the matrix."""
    plt.figure(figsize=(10, 8))
    
    # Plot the matrix
    plt.subplot(2, 2, 1)
    plt.imshow(matrix.cpu().numpy(), cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title(f'{title} - Full Matrix')
    
    # Plot diagonal elements
    plt.subplot(2, 2, 2)
    diagonal = torch.diag(matrix).cpu().numpy()
    plt.plot(diagonal, 'b-', alpha=0.7, label='Diagonal elements')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='y=1 (identity)')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Diagonal Elements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot eigenvalue distribution
    plt.subplot(2, 2, 3)
    eigenvals = torch.linalg.eigvals(matrix).real.cpu().numpy()
    plt.hist(eigenvals, bins=30, alpha=0.7, color='green')
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='λ=1 (identity)')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Count')
    plt.title('Eigenvalue Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot difference from identity
    plt.subplot(2, 2, 4)
    n_embd = matrix.shape[0]
    identity = torch.eye(n_embd, device=matrix.device, dtype=matrix.dtype)
    diff = (matrix - identity).cpu().numpy()
    plt.imshow(diff, cmap='RdBu_r', aspect='auto')
    plt.colorbar()
    plt.title('Difference from Identity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def analyze_checkpoint(checkpoint_path, output_dir="analysis_output"):
    """
    Main analysis function for a checkpoint - using check.py loading method.
    """
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load checkpoint using check.py method
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if checkpoint IS the model state dict (keys start with 'module.')
    first_key = list(checkpoint.keys())[0] if checkpoint else ""
    
    if first_key.startswith('module.'):
        # Checkpoint IS the model state dict
        print("✅ Checkpoint is model state dict directly")
        model_state_dict = checkpoint
    else:
        # Normal checkpoint format - find model state dict
        model_state_key = None
        possible_keys = ['model_state_dict', 'model', 'state_dict', 'net']
        
        for key in possible_keys:
            if key in checkpoint:
                model_state_key = key
                break
        
        if model_state_key is None:
            print(f"❌ No model state dict found. Available keys: {list(checkpoint.keys())}")
            return
        
        print(f"✅ Found model state at key: '{model_state_key}'")
        model_state_dict = checkpoint[model_state_key]
    
    # Clean keys (remove 'module.' prefix if present)
    clean_state_dict = {}
    for key, value in model_state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        clean_state_dict[new_key] = value
    
    print(f"🔧 Cleaned {len(clean_state_dict)} parameter keys")
    
    # Detect model dimensions using check.py method
    wte_key = None
    for key in clean_state_dict.keys():
        if 'wte.weight' in key or 'token_embedding' in key:
            wte_key = key
            break
    
    if wte_key is None:
        print("Error: Could not find embedding weights in checkpoint!")
        return
    
    n_embd = clean_state_dict[wte_key].shape[1]
    vocab_size = clean_state_dict[wte_key].shape[0]
    
    # Infer number of heads (common sizes)
    for head_size in [64, 32, 128, 16]:
        if n_embd % head_size == 0:
            n_head = n_embd // head_size
            head_dim = head_size
            break
    else:
        # Default fallback
        n_head = 8
        head_dim = n_embd // n_head
    
    # Count layers
    layer_count = 0
    for key in clean_state_dict.keys():
        if 'transformer.h.' in key:
            layer_num = int(key.split('transformer.h.')[1].split('.')[0])
            layer_count = max(layer_count, layer_num + 1)
    
    print(f"Detected: {layer_count} layers, n_embd={n_embd}, n_head={n_head}, head_dim={head_dim}")
    
    # Analyze each layer
    all_results = []
    
    for layer_idx in range(layer_count):
        print(f"\\nAnalyzing layer {layer_idx}...")
        
        try:
            # Extract V and W_O matrices
            W_V, W_O = extract_v_wo_matrices(clean_state_dict, layer_idx, n_head, head_dim)
            
            # Compute V * W_O
            V_W_O = W_V @ W_O
            
            # Analyze similarity to identity
            results = analyze_identity_similarity(V_W_O, layer_idx)
            results['layer'] = layer_idx
            all_results.append(results)
            
            print(f"  Frobenius distance to identity: {results['frobenius_distance']:.4f}")
            print(f"  Normalized Frobenius distance: {results['normalized_frobenius']:.4f}")
            print(f"  Diagonal dominance ratio: {results['diagonal_to_off_diagonal_ratio']:.4f}")
            print(f"  Eigenvalues close to 1: {results['eigenvalues_close_to_one']:.2%}")
            
            # Create visualization for first and last layers
            if layer_idx == 0 or layer_idx == layer_count - 1:
                save_path = f"{output_dir}/layer_{layer_idx}_V_W_O_analysis.png"
                visualize_matrix(V_W_O, f"Layer {layer_idx}: V*W_O", save_path)
            
        except Exception as e:
            print(f"  Error analyzing layer {layer_idx}: {e}")
    
    # Summary analysis
    print(f"\\n{'='*50}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*50}")
    
    if all_results:
        # Compute statistics across layers
        frobenius_distances = [r['frobenius_distance'] for r in all_results]
        normalized_distances = [r['normalized_frobenius'] for r in all_results]
        diagonal_ratios = [r['diagonal_to_off_diagonal_ratio'] for r in all_results]
        eigenvalue_ratios = [r['eigenvalues_close_to_one'] for r in all_results]
        
        print(f"Frobenius distance to identity:")
        print(f"  Mean: {np.mean(frobenius_distances):.4f} ± {np.std(frobenius_distances):.4f}")
        print(f"  Range: [{np.min(frobenius_distances):.4f}, {np.max(frobenius_distances):.4f}]")
        
        print(f"\\nNormalized Frobenius distance:")
        print(f"  Mean: {np.mean(normalized_distances):.4f} ± {np.std(normalized_distances):.4f}")
        
        print(f"\\nDiagonal dominance ratio:")
        print(f"  Mean: {np.mean(diagonal_ratios):.4f} ± {np.std(diagonal_ratios):.4f}")
        
        print(f"\\nEigenvalues close to 1:")
        print(f"  Mean: {np.mean(eigenvalue_ratios):.2%} ± {np.std(eigenvalue_ratios):.2%}")
        
        # Determine if V*W_O is close to identity
        avg_normalized_distance = np.mean(normalized_distances)
        avg_diagonal_ratio = np.mean(diagonal_ratios)
        avg_eigenvalue_ratio = np.mean(eigenvalue_ratios)
        
        print(f"\\n{'='*50}")
        print("CONCLUSION")
        print(f"{'='*50}")
        
        if avg_normalized_distance < 0.5 and avg_diagonal_ratio > 2.0 and avg_eigenvalue_ratio > 0.5:
            print("✅ V*W_O appears to be CLOSE TO IDENTITY!")
            print("   This supports the research finding that attention is primarily routing.")
        elif avg_normalized_distance < 1.0 and avg_diagonal_ratio > 1.5:
            print("⚠️  V*W_O shows SOME identity-like behavior.")
            print("   Attention may be partially routing-based.")
        else:
            print("❌ V*W_O does NOT appear close to identity.")
            print("   Attention performs significant transformation, not just routing.")
        
        # Save summary results
        summary_path = f"{output_dir}/summary_results.txt"
        with open(summary_path, 'w') as f:
            f.write("V*W_O Identity Analysis Summary\\n")
            f.write("="*50 + "\\n")
            f.write(f"Checkpoint: {checkpoint_path}\\n")
            f.write(f"Model: {layer_count} layers, {n_embd} embedding dim, {n_head} heads\\n\\n")
            
            for result in all_results:
                f.write(f"Layer {result['layer']}:\\n")
                f.write(f"  Frobenius distance: {result['frobenius_distance']:.4f}\\n")
                f.write(f"  Normalized distance: {result['normalized_frobenius']:.4f}\\n")
                f.write(f"  Diagonal ratio: {result['diagonal_to_off_diagonal_ratio']:.4f}\\n")
                f.write(f"  Eigenvalues near 1: {result['eigenvalues_close_to_one']:.2%}\\n\\n")
        
        print(f"\\nDetailed results saved to: {summary_path}")

if __name__ == "__main__":
    # Example usage - modify checkpoint path as needed
    checkpoint_path = "outputs/vanilla_model.pt"  # Adjust this path
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please provide the correct path to your vanilla transformer checkpoint.")
        print("Usage: python analyze_v_wo_identity.py <checkpoint_path>")
        sys.exit(1)
    
    analyze_checkpoint(checkpoint_path)