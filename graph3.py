#!/usr/bin/env python
"""
Process .pt validation checkpoints and create validation performance graphs.
Simple implementation for .pt files only.
"""

import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import math

def is_nan(x):
    """Simple NaN check."""
    if isinstance(x, (int, float)):
        return math.isnan(x) or x != x
    return False

def extract_checkpoint_metrics(checkpoint_path):
    """Extract validation metrics from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        metrics = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': checkpoint.get('epoch', 0),
            'global_batch': checkpoint.get('global_batch', 0),
            'train_loss': checkpoint.get('loss', float('nan')),
            'val_loss': checkpoint.get('val_loss', float('nan')),
            'val_perplexity': checkpoint.get('val_perplexity', float('nan'))
        }
        
        return metrics
    except Exception as e:
        print(f"Error reading {checkpoint_path}: {e}")
        return None

def create_validation_graphs(output_dir):
    """Create validation performance graphs from .pt checkpoint files."""
    
    print(f"📊 Processing .pt files from: {output_dir}")
    
    # Find .pt checkpoint files
    checkpoint_patterns = [
        '*.pt',
        'checkpoint_*.pt', 
        'checkpoints/*.pt'
    ]
    
    checkpoint_metrics = []
    for pattern in checkpoint_patterns:
        checkpoint_files = list(Path(output_dir).glob(pattern))
        for cp_file in checkpoint_files:
            print(f"Processing: {cp_file.name}")
            metrics = extract_checkpoint_metrics(str(cp_file))
            if metrics:
                checkpoint_metrics.append(metrics)
    
    print(f"Found {len(checkpoint_metrics)} checkpoint files with validation data")
    
    if not checkpoint_metrics:
        print("❌ No validation data found in checkpoint files")
        return
    
    # Create graphs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Validation Performance Analysis', fontsize=16)
    
    # Graph 1: Validation Loss over Epochs
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    
    if epochs and val_losses:
        axes[0,0].plot(epochs, val_losses, 'bo-', label='Validation Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Validation Loss')
        axes[0,0].set_title('Validation Loss vs Epoch')
        axes[0,0].grid(True)
        axes[0,0].legend()
    
    # Graph 2: Validation Perplexity over Epochs
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['val_perplexity'])]
    val_ppls = [m['val_perplexity'] for m in checkpoint_metrics if not is_nan(m['val_perplexity'])]
    
    if epochs and val_ppls:
        axes[0,1].plot(epochs, val_ppls, 'ro-', label='Validation Perplexity')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Validation Perplexity')
        axes[0,1].set_title('Validation Perplexity vs Epoch')
        axes[0,1].grid(True)
        axes[0,1].legend()
    
    # Graph 3: Validation Loss over Global Batches
    batches = [m['global_batch'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    
    if batches and val_losses:
        axes[1,0].plot(batches, val_losses, 'go-', label='Validation Loss')
        axes[1,0].set_xlabel('Global Batch')
        axes[1,0].set_ylabel('Validation Loss')
        axes[1,0].set_title('Validation Loss vs Global Batch')
        axes[1,0].grid(True)
        axes[1,0].legend()
    
    # Graph 4: Train vs Validation Loss
    train_losses = [m['train_loss'] for m in checkpoint_metrics if not is_nan(m['train_loss'])]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['train_loss']) and not is_nan(m['val_loss'])]
    
    if epochs and train_losses and val_losses:
        axes[1,1].plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
        axes[1,1].plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_title('Train vs Validation Loss')
        axes[1,1].grid(True)
        axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save graph
    graph_path = os.path.join(output_dir, 'validation_analysis.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved validation graph: {graph_path}")
    
    # Print summary
    print(f"\n📋 VALIDATION SUMMARY:")
    sorted_checkpoints = sorted(checkpoint_metrics, key=lambda x: x.get('val_loss', float('inf')))
    best = sorted_checkpoints[0] if sorted_checkpoints else None
    
    if best:
        print(f"🏆 Best checkpoint: {best['checkpoint']}")
        print(f"   Epoch: {best['epoch']}")
        print(f"   Val Loss: {best['val_loss']:.4f}")
        print(f"   Val Perplexity: {best['val_perplexity']:.2f}")
    
    print(f"\n📊 All checkpoints:")
    for i, cp in enumerate(sorted_checkpoints[:5]):  # Show top 5
        marker = "🏆" if i == 0 else f"{i+1}."
        print(f"{marker} {cp['checkpoint']} - Val Loss: {cp['val_loss']:.4f}, PPL: {cp['val_perplexity']:.2f}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create validation performance graphs')
    parser.add_argument('--output_dir', type=str, default='./outputs/sym_4gpu_simple',
                       help='Directory containing checkpoints and logs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Directory not found: {args.output_dir}")
        return
    create_validation_graphs(args.output_dir)

if __name__ == "__main__":
    main()