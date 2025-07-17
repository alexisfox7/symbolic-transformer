#!/usr/bin/env python3
"""
Script to plot training loss over batches (not epochs) from JSON logs.
Can optionally compare two different training runs.
"""

import json
import matplotlib.pyplot as plt
import sys
import os
from glob import glob
import numpy as np

def read_batch_data(log_file):
    """Read batch data from a single log file."""
    batches = []
    losses = []
    epochs = []
    perplexities = []
    
    epoch_max_batches = {}  # Track max batch number per epoch
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Track batch events
                if data.get('event') == 'batch':
                    epoch = data.get('epoch', 0)
                    batch = data.get('batch', 0)
                    loss = data.get('loss')
                    perplexity = data.get('perplexity')
                    
                    if loss is not None:
                        # Track maximum batch number for each epoch
                        if epoch not in epoch_max_batches or batch > epoch_max_batches[epoch]:
                            epoch_max_batches[epoch] = batch
                        
                        # Calculate global batch number
                        global_batch = batch
                        for e in range(1, epoch):
                            if e in epoch_max_batches:
                                global_batch += epoch_max_batches[e] + 1
                        
                        batches.append(global_batch)
                        losses.append(loss)
                        epochs.append(epoch)
                        if perplexity is not None:
                            perplexities.append(perplexity)
                        
            except json.JSONDecodeError:
                continue
    
    return batches, losses, epochs, perplexities

def plot_batch_loss(log_dir, compare_dir=None):
    """Plot training loss over batches from JSON logs."""
    # Find the most recent training log
    log_files = sorted(glob(os.path.join(log_dir, "logs", "training_*.jsonl")))
    
    if not log_files:
        print(f"No training logs found in {log_dir}/logs/")
        return
    
    log_file = log_files[-1]  # Use most recent
    print(f"Reading log file: {log_file}")
    
    # Parse the JSON lines for batch data
    batches = []
    losses = []
    epochs = []
    perplexities = []
    
    epoch_max_batches = {}  # Track max batch number per epoch
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Track batch events
                if data.get('event') == 'batch':
                    epoch = data.get('epoch', 0)
                    batch = data.get('batch', 0)
                    loss = data.get('loss')
                    perplexity = data.get('perplexity')
                    
                    if loss is not None:
                        # Track maximum batch number for each epoch
                        if epoch not in epoch_max_batches or batch > epoch_max_batches[epoch]:
                            epoch_max_batches[epoch] = batch
                        
                        # Calculate global batch number
                        global_batch = batch
                        for e in range(1, epoch):
                            if e in epoch_max_batches:
                                global_batch += epoch_max_batches[e] + 1
                        
                        batches.append(global_batch)
                        losses.append(loss)
                        epochs.append(epoch)
                        if perplexity is not None:
                            perplexities.append(perplexity)
                        
            except json.JSONDecodeError:
                continue
    
    if not batches:
        print("No batch data found in log file")
        return
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Loss over batches
    ax1.plot(batches, losses, 'b-', linewidth=1, alpha=0.8)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Batches')
    ax1.grid(True, alpha=0.3)
    
    # Add epoch boundaries as vertical lines
    epoch_changes = []
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_changes.append(batches[i])
            ax1.axvline(x=batches[i], color='gray', linestyle='--', alpha=0.5)
            ax1.text(batches[i], ax1.get_ylim()[1]*0.95, f'Epoch {epochs[i]}', 
                    rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)
    
    # Plot 2: Perplexity over batches (if available)
    if perplexities:
        ax2.plot(batches[:len(perplexities)], perplexities, 'g-', linewidth=1, alpha=0.8)
        ax2.set_ylabel('Perplexity')
        ax2.set_xlabel('Batch')
        ax2.set_title('Perplexity Over Batches')
        ax2.grid(True, alpha=0.3)
        
        # Add epoch boundaries
        for batch_num in epoch_changes:
            ax2.axvline(x=batch_num, color='gray', linestyle='--', alpha=0.5)
        
        # Use log scale for perplexity if values vary widely
        if max(perplexities) / min(perplexities) > 100:
            ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_batch_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary (Batch-level):")
    print(f"  Total batches: {len(batches)}")
    print(f"  Total epochs: {max(epochs) if epochs else 0}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Min loss: {min(losses):.4f} (batch {batches[losses.index(min(losses))]})")
    print(f"  Max loss: {max(losses):.4f} (batch {batches[losses.index(max(losses))]})")
    
    if perplexities:
        print(f"\n  Initial perplexity: {perplexities[0]:.2f}")
        print(f"  Final perplexity: {perplexities[-1]:.2f}")
        print(f"  Min perplexity: {min(perplexities):.2f}")
        print(f"  Max perplexity: {max(perplexities):.2f}")
    
    # Calculate and print smoothed statistics
    if len(losses) > 100:
        window_size = max(10, len(losses) // 50)
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        print(f"\n  Smoothed final loss (window={window_size}): {smoothed_losses[-1]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "outputs/cascade_kronecker_reason"
    
    plot_batch_loss(log_dir)