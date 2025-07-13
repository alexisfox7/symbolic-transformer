#!/usr/bin/env python3
"""
Simple script to plot training loss from JSON logs.
"""

import json
import matplotlib.pyplot as plt
import sys
import os
from glob import glob

def plot_training_loss(log_dir):
    """Plot training loss from JSON logs."""
    # Find the most recent training log
    log_files = sorted(glob(os.path.join(log_dir, "logs", "training_*.jsonl")))
    
    if not log_files:
        print(f"No training logs found in {log_dir}/logs/")
        return
    
    log_file = log_files[-1]  # Use most recent
    print(f"Reading log file: {log_file}")
    
    # Parse the JSON lines
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('event') == 'epoch_end':
                    epoch = data.get('epoch')
                    loss = data.get('loss')
                    val_loss = data.get('val_loss')
                    
                    if epoch is not None and loss is not None:
                        epochs.append(epoch)
                        train_losses.append(loss)
                        if val_loss is not None:
                            val_losses.append(val_loss)
            except json.JSONDecodeError:
                continue
    
    if not epochs:
        print("No epoch data found in log file")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    # Plot validation loss if available
    if val_losses and len(val_losses) == len(epochs):
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(log_dir, 'training_loss_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(epochs)}")
    print(f"  Final training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  Final validation loss: {val_losses[-1]:.4f}")
        print(f"  Best validation loss: {min(val_losses):.4f} (epoch {epochs[val_losses.index(min(val_losses))]})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "outputs/cascade_sparse"
    
    plot_training_loss(log_dir)