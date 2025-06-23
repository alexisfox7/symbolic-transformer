#!/usr/bin/env python3
"""
Simple script to get loss from checkpoint.
"""

import torch
import sys

def get_loss(checkpoint_path):
    """Get loss from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    # Main loss
    if 'loss' in checkpoint:
        print(f"Training Loss: {checkpoint['loss']:.4f}")
    
    # Validation loss
    if 'val_loss' in checkpoint:
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Loss history
    if 'training_metrics' in checkpoint and 'epoch_losses' in checkpoint['training_metrics']:
        losses = checkpoint['training_metrics']['epoch_losses']
        print(f"Loss History: {[round(l, 4) for l in losses]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_loss.py checkpoint.pt")
        sys.exit(1)
    
    get_loss(sys.argv[1])