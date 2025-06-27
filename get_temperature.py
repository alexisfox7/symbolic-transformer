#!/usr/bin/env python3
"""
Script to extract and display learned temperature parameters from checkpoint.
"""

import torch
import sys

def get_temperatures(checkpoint_path):
    """Extract temperature parameters from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print("=" * 50)
    
    # Get model state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
    
    # Find all temperature parameters
    temperatures = {}
    for name, param in state_dict.items():
        if 'temperature' in name:
            temperatures[name] = param.item() if param.numel() == 1 else param.tolist()
    
    if temperatures:
        print("Learned Temperature Values:")
        for name, value in sorted(temperatures.items()):
            # Extract layer info from parameter name
            parts = name.split('.')
            layer_info = []
            for i, part in enumerate(parts):
                if part == 'h' and i + 1 < len(parts):
                    layer_info.append(f"Layer {parts[i+1]}")
                elif part == 'attn':
                    layer_info.append("Attention")
            
            layer_str = " - ".join(layer_info) if layer_info else name
            print(f"  {layer_str}: {value:.4f}")
    else:
        print("No temperature parameters found in checkpoint.")
        print("Make sure the model was trained with --learnable_temperature flag.")
    
    print("=" * 50)
    
    # Show some statistics if multiple temperatures
    if len(temperatures) > 1:
        values = list(temperatures.values())
        print(f"Temperature Statistics:")
        print(f"  Mean: {sum(values)/len(values):.4f}")
        print(f"  Min:  {min(values):.4f}")
        print(f"  Max:  {max(values):.4f}")
        print(f"  Std:  {torch.tensor(values).std().item():.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_temperature.py checkpoint.pt")
        sys.exit(1)
    
    get_temperatures(sys.argv[1])