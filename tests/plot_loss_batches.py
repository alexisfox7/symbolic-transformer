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
    # Read primary data
    log_files = sorted(glob(os.path.join(log_dir, "logs", "training_*.jsonl")))
    
    if not log_files:
        print(f"No training logs found in {log_dir}/logs/")
        return
    
    log_file = log_files[-1]  # Use most recent
    print(f"Reading primary log file: {log_file}")
    
    batches, losses, epochs, perplexities = read_batch_data(log_file)
    
    if not batches:
        print("No batch data found in primary log file")
        return
    
    # Read comparison data if provided
    compare_data = None
    if compare_dir:
        compare_log_files = sorted(glob(os.path.join(compare_dir, "logs", "training_*.jsonl")))
        if compare_log_files:
            compare_log_file = compare_log_files[-1]
            print(f"Reading comparison log file: {compare_log_file}")
            compare_batches, compare_losses, compare_epochs, compare_perplexities = read_batch_data(compare_log_file)
            if compare_batches:
                compare_data = {
                    'batches': compare_batches,
                    'losses': compare_losses,
                    'epochs': compare_epochs,
                    'perplexities': compare_perplexities,
                    'name': os.path.basename(compare_dir)
                }
        else:
            print(f"No training logs found in {compare_dir}/logs/")
    
    primary_name = os.path.basename(log_dir)
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Loss over batches
    ax1.plot(batches, losses, 'b-', linewidth=1.5, alpha=0.8, label=primary_name)
    
    # Add comparison if available
    if compare_data:
        ax1.plot(compare_data['batches'], compare_data['losses'], 'r-', 
                linewidth=1.5, alpha=0.8, label=compare_data['name'])
    
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Batches')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add epoch boundaries as vertical lines for primary data
    epoch_changes = []
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_changes.append(batches[i])
            ax1.axvline(x=batches[i], color='gray', linestyle='--', alpha=0.3)
            ax1.text(batches[i], ax1.get_ylim()[1]*0.95, f'Epoch {epochs[i]}', 
                    rotation=90, va='top', ha='right', fontsize=8, alpha=0.5)
    
    # Plot 2: Perplexity over batches (if available)
    has_perplexity = False
    if perplexities:
        ax2.plot(batches[:len(perplexities)], perplexities, 'b-', 
                linewidth=1.5, alpha=0.8, label=primary_name)
        has_perplexity = True
        
    if compare_data and compare_data['perplexities']:
        ax2.plot(compare_data['batches'][:len(compare_data['perplexities'])], 
                compare_data['perplexities'], 'r-', linewidth=1.5, alpha=0.8, 
                label=compare_data['name'])
        has_perplexity = True
    
    if has_perplexity:
        ax2.set_ylabel('Perplexity')
        ax2.set_xlabel('Batch')
        ax2.set_title('Perplexity Over Batches')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add epoch boundaries
        for batch_num in epoch_changes:
            ax2.axvline(x=batch_num, color='gray', linestyle='--', alpha=0.3)
        
        # Use log scale for perplexity if values vary widely
        all_perplexities = perplexities if perplexities else []
        if compare_data and compare_data['perplexities']:
            all_perplexities = all_perplexities + compare_data['perplexities']
        if all_perplexities and max(all_perplexities) / min(all_perplexities) > 100:
            ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_batch_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary for {primary_name} (Batch-level):")
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
    
    # Print comparison statistics if available
    if compare_data:
        print(f"\nTraining Summary for {compare_data['name']} (Batch-level):")
        print(f"  Total batches: {len(compare_data['batches'])}")
        print(f"  Total epochs: {max(compare_data['epochs']) if compare_data['epochs'] else 0}")
        print(f"  Initial loss: {compare_data['losses'][0]:.4f}")
        print(f"  Final loss: {compare_data['losses'][-1]:.4f}")
        print(f"  Min loss: {min(compare_data['losses']):.4f}")
        
        if compare_data['perplexities']:
            print(f"\n  Initial perplexity: {compare_data['perplexities'][0]:.2f}")
            print(f"  Final perplexity: {compare_data['perplexities'][-1]:.2f}")
            print(f"  Min perplexity: {min(compare_data['perplexities']):.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        log_dir = sys.argv[1]
        compare_dir = sys.argv[2]
        plot_batch_loss(log_dir, compare_dir)
    elif len(sys.argv) > 1:
        log_dir = sys.argv[1]
        plot_batch_loss(log_dir)
    else:
        log_dir = "outputs/cascade_kronecker_reason"
        plot_batch_loss(log_dir)