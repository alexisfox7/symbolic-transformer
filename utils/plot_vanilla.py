#!/usr/bin/env python
# plot_vanilla_training.py
"""
Quick script to plot training loss for vanilla transformer.
"""

import sys
import os

# Add utils to path if needed
if 'utils' not in sys.path:
    sys.path.append('utils')

# Import the plotting function
from plot_training import plot_training_loss

def main():
    # Adjust this path to match your vanilla training output directory
    vanilla_output_dir = "./outputs/vanilla_baseline"  # Default from training script
    
    # Alternative paths to check
    possible_dirs = [
        "./outputs/vanilla_baseline",
        "./outputs/vanilla_test", 
        "./outputs/vanilla",
        "./vanilla_output"
    ]
    
    # Find the actual output directory
    output_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            output_dir = dir_path
            print(f"Found output directory: {output_dir}")
            break
    
    if output_dir is None:
        print("Could not find vanilla training output directory.")
        print("Please check these locations:")
        for dir_path in possible_dirs:
            print(f"  {dir_path}")
        print("\nOr specify the correct path in this script.")
        return
    
    # Create the plot
    plot_training_loss(
        output_dir=output_dir,
        model_name="Vanilla Transformer",
        save_plot=True,
        show_plot=True,
        figsize=(10, 6)
    )
    
    print(f"\nPlot created from data in: {output_dir}")

if __name__ == "__main__":
    main()