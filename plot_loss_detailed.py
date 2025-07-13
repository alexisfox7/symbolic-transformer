#!/usr/bin/env python3
"""
Detailed loss plotting with batch-level granularity.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from glob import glob

def plot_detailed_loss(log_dir, show_batches=False):
    """Plot training loss with optional batch-level detail."""
    # Find the most recent training log
    log_files = sorted(glob(os.path.join(log_dir, "logs", "training_*.jsonl")))
    
    if not log_files:
        print(f"No training logs found in {log_dir}/logs/")
        return
    
    log_file = log_files[-1]
    print(f"Reading log file: {log_file}")
    
    # Parse the JSON lines
    epoch_data = {}
    batch_data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                if data.get('event') == 'epoch_end':
                    epoch = data.get('epoch')
                    if epoch is not None:
                        epoch_data[epoch] = {
                            'loss': data.get('loss'),
                            'val_loss': data.get('val_loss'),
                            'perplexity': data.get('perplexity'),
                            'val_perplexity': data.get('val_perplexity')
                        }
                
                elif data.get('event') == 'batch' and show_batches:
                    batch_data.append({
                        'epoch': data.get('epoch', 0),
                        'batch': data.get('batch'),
                        'loss': data.get('loss')
                    })
                    
            except json.JSONDecodeError:
                continue
    
    if not epoch_data:
        print("No epoch data found in log file")
        return
    
    # Prepare epoch data
    epochs = sorted(epoch_data.keys())
    train_losses = [epoch_data[e]['loss'] for e in epochs if epoch_data[e]['loss'] is not None]
    val_losses = [epoch_data[e]['val_loss'] for e in epochs if epoch_data[e].get('val_loss') is not None]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1.plot(epochs[:len(train_losses)], train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Add batch losses if requested
    if show_batches and batch_data:
        batch_epochs = [b['epoch'] + b['batch']/max(b['batch'] for b in batch_data if b['epoch'] == 0) 
                       for b in batch_data]
        batch_losses = [b['loss'] for b in batch_data]
        ax1.plot(batch_epochs, batch_losses, 'lightblue', alpha=0.5, linewidth=0.5, label='Batch Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    train_perps = [epoch_data[e]['perplexity'] for e in epochs 
                   if epoch_data[e].get('perplexity') is not None]
    val_perps = [epoch_data[e]['val_perplexity'] for e in epochs 
                 if epoch_data[e].get('val_perplexity') is not None]
    
    if train_perps:
        ax2.plot(epochs[:len(train_perps)], train_perps, 'b-', label='Training Perplexity', linewidth=2)
    if val_perps:
        ax2.plot(epochs[:len(val_perps)], val_perps, 'r-', label='Validation Perplexity', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_detailed_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print detailed summary
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Final training loss: {train_losses[-1]:.4f}")
    print(f"  Final training perplexity: {train_perps[-1]:.2f}" if train_perps else "")
    
    if val_losses:
        best_val_idx = np.argmin(val_losses)
        print(f"  Final validation loss: {val_losses[-1]:.4f}")
        print(f"  Best validation loss: {val_losses[best_val_idx]:.4f} (epoch {epochs[best_val_idx]})")
    
    # Check for loss trends
    if len(train_losses) > 3:
        recent_trend = np.mean(np.diff(train_losses[-3:]))
        if recent_trend > 0:
            print("  ⚠️  Warning: Training loss is increasing in recent epochs")
        elif abs(recent_trend) < 0.001:
            print("  ℹ️  Note: Training loss has plateaued")

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/cascade_sparse"
    show_batches = "--batches" in sys.argv
    
    plot_detailed_loss(log_dir, show_batches)