#!/usr/bin/env python3
"""
Simple script to get loss and perplexity from checkpoint.
"""

import torch
import sys
import math

def get_loss(checkpoint_path):
    """Get loss and perplexity from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    # Main loss and perplexity
    if 'loss' in checkpoint:
        loss = checkpoint['loss']
        perplexity = math.exp(loss)
        print(f"Training Loss: {loss:.4f}")
        print(f"Training Perplexity: {perplexity:.2f}")
    
    # Validation loss and perplexity
    if 'val_loss' in checkpoint:
        val_loss = checkpoint['val_loss']
        val_perplexity = math.exp(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Perplexity: {val_perplexity:.2f}")
    
    # Loss history with perplexities
    if 'training_metrics' in checkpoint and 'epoch_losses' in checkpoint['training_metrics']:
        losses = checkpoint['training_metrics']['epoch_losses']
        perplexities = [math.exp(l) for l in losses]
        print(f"Loss History: {[round(l, 4) for l in losses]}")
        print(f"Perplexity History: {[round(p, 2) for p in perplexities]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_loss.py checkpoint.pt")
        sys.exit(1)
    
    get_loss(sys.argv[1])