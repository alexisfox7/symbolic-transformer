#!/usr/bin/env python3
"""
Simple script to extract and display config from checkpoint.
"""

import torch
import sys
import json
from pathlib import Path

def get_config(checkpoint_path):
    """Extract config from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    # Look for config in different possible keys
    config = None
    config_keys = ['config', 'model_config', 'args', 'model_args']
    
    for key in config_keys:
        if key in checkpoint:
            config = checkpoint[key]
            print(f"Found config under key: '{key}'")
            break
    
    # Print max_samples from config if available
    if config and hasattr(config, 'max_samples'):
        print(f"Max samples: {config.max_samples:,}")
    elif config and isinstance(config, dict) and 'max_samples' in config:
        print(f"Max samples: {config['max_samples']:,}")
    
    print()
    
    if config is None:
        print("No config found in checkpoint")
        print("Available keys:", list(checkpoint.keys()))
        return None
    
    # Display config
    print("\nModel Configuration:")
    print("=" * 50)
    
    if hasattr(config, '__dict__'):
        # Config object with attributes
        config_dict = vars(config)
    elif isinstance(config, dict):
        # Config is already a dictionary
        config_dict = config
    else:
        print(f"Config type: {type(config)}")
        print(config)
        return config
    
    # Pretty print the config
    for key, value in sorted(config_dict.items()):
        if not key.startswith('_'):  # Skip private attributes
            print(f"{key:25}: {value}")
    
    return config

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_config.py <checkpoint_path>")
        print("\nExample:")
        print("python get_config.py outputs/checkpoints/cascade/checkpoint_epoch_1.pt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    try:
        config = get_config(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()