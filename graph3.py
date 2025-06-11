#!/usr/bin/env python3
"""
Quick script to check if V*W_O ≈ Identity in your vanilla transformer.
Run this first to get a quick answer!
"""

import torch
import numpy as np
import os
import glob
import sys

def quick_identity_check(checkpoint_path):
    """Quick check without visualizations - using check.py loading method."""
    print(f"Loading: {checkpoint_path}")
    
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
    
    # Find model dimensions using check.py method
    wte_key = None
    for key in clean_state_dict.keys():
        if 'wte.weight' in key or 'token_embedding' in key:
            wte_key = key
            break
    
    if wte_key is None:
        print("❌ Could not find embedding weights!")
        return
    
    n_embd = clean_state_dict[wte_key].shape[1]
    vocab_size = clean_state_dict[wte_key].shape[0]
    
    # Detect number of heads (check.py style)
    n_head = 8  # default
    for head_size in [64, 32, 128, 16]:
        if n_embd % head_size == 0:
            n_head = n_embd // head_size
            break
    
    # Count layers
    layer_count = 0
    for key in clean_state_dict.keys():
        if 'transformer.h.' in key:
            layer_num = int(key.split('transformer.h.')[1].split('.')[0])
            layer_count = max(layer_count, layer_num + 1)
    
    print(f"Model: {layer_count} layers, {n_embd} embedding dim")
    
    # Quick analysis of a few layers
    distances = []
    diagonal_ratios = []
    
    for layer_idx in [0, layer_count//2, layer_count-1]:  # First, middle, last
        try:
            # Extract matrices
            c_attn_weight = clean_state_dict[f'transformer.h.{layer_idx}.attn.c_attn.weight']
            c_proj_weight = clean_state_dict[f'transformer.h.{layer_idx}.attn.c_proj.weight']
            
            # Get V and W_O
            W_V = c_attn_weight[2*n_embd:3*n_embd, :].T  # V projection
            W_O = c_proj_weight.T  # Output projection
            
            # Compute V * W_O
            V_W_O = W_V @ W_O
            
            # Quick metrics
            identity = torch.eye(n_embd)
            frobenius_dist = torch.norm(V_W_O - identity, p='fro').item()
            normalized_dist = frobenius_dist / torch.norm(identity, p='fro').item()
            
            # Diagonal dominance
            diagonal = torch.diag(V_W_O)
            off_diagonal = V_W_O[~torch.eye(n_embd, dtype=bool)]
            diagonal_ratio = torch.abs(diagonal).mean() / (torch.abs(off_diagonal).mean() + 1e-8)
            
            distances.append(normalized_dist)
            diagonal_ratios.append(diagonal_ratio.item())
            
            print(f"Layer {layer_idx}: distance={normalized_dist:.3f}, diagonal_ratio={diagonal_ratio:.2f}")
            
        except Exception as e:
            print(f"Error on layer {layer_idx}: {e}")
    
    if distances:
        avg_distance = np.mean(distances)
        avg_ratio = np.mean(diagonal_ratios)
        
        print(f"\\n{'='*50}")
        print(f"QUICK RESULT:")
        print(f"Average distance from identity: {avg_distance:.3f}")
        print(f"Average diagonal dominance: {avg_ratio:.2f}")
        
        if avg_distance < 0.5 and avg_ratio > 2.0:
            print("✅ YES! V*W_O appears close to identity")
            print("   Your transformer does mostly routing!")
        elif avg_distance < 1.0:
            print("⚠️  PARTIALLY - some identity-like behavior")
            print("   Mixed routing and transformation")
        else:
            print("❌ NO - V*W_O is not identity-like")
            print("   Your transformer does significant transformation")

def find_checkpoints():
    """Find potential checkpoint files using check.py patterns."""
    patterns = [
        "outputs/*.pt",
        "outputs/*/*.pt", 
        "outputs/*/*/*.pt",
        "checkpoints/*.pt",
        "*.pt"
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    # Filter out obviously non-model files
    model_checkpoints = []
    for cp in all_checkpoints:
        filename = os.path.basename(cp).lower()
        # Skip if it looks like optimizer or config files
        if any(skip in filename for skip in ['optimizer', 'config', 'metadata']):
            continue
        model_checkpoints.append(cp)
    
    return model_checkpoints

if __name__ == "__main__":
    print("🔍 Looking for transformer checkpoints...")
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if os.path.exists(checkpoint_path):
            print(f"🎯 Using provided checkpoint: {checkpoint_path}")
            quick_identity_check(checkpoint_path)
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
        exit()
    
    # Otherwise search for checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("❌ No .pt files found!")
        print("Please make sure you have a trained transformer checkpoint.")
        print("Usage: python quick_check_identity.py <checkpoint_path>")
        exit(1)
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for i, cp in enumerate(checkpoints):
        print(f"  {i+1}. {cp}")
    
    # Try to find vanilla transformer checkpoint
    vanilla_checkpoints = [cp for cp in checkpoints if 'vanilla' in cp.lower()]
    
    if vanilla_checkpoints:
        print(f"\\n🎯 Found vanilla checkpoint: {vanilla_checkpoints[0]}")
        quick_identity_check(vanilla_checkpoints[0])
    else:
        print(f"\\n🤔 No obvious vanilla checkpoint found.")
        print(f"Trying first checkpoint: {checkpoints[0]}")
        quick_identity_check(checkpoints[0])