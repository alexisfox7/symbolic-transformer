#!/usr/bin/env python
# utils/plot_training.py
"""
Training visualization utilities for Symbolic and Vanilla Transformers.
Provides functions to plot training curves from logs and checkpoints.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import torch
import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def extract_loss_from_logs(log_file_path: str) -> List[Tuple[int, float]]:
    """
    Extract epoch and loss information from training log files.
    
    Args:
        log_file_path: Path to the training log file
        
    Returns:
        List of (epoch, loss) tuples
    """
    epoch_losses = []
    
    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return epoch_losses
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for epoch completion lines
            # Format: "Epoch X completed, Avg Training Loss: Y.YYYY"
            match = re.search(r'Epoch (\d+) completed, Avg Training Loss: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epoch_losses.append((epoch, loss))
    
    return epoch_losses

def extract_loss_from_checkpoints(checkpoint_dir: str) -> List[Tuple[int, float]]:
    """
    Extract loss information from checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        List of (epoch, loss) tuples
    """
    epoch_losses = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return epoch_losses
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pt'):
            checkpoint_files.append(file)
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)
            
            if loss is not None:
                epoch_losses.append((epoch, loss))
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_file}: {e}")
            continue
    
    return epoch_losses

def plot_training_loss(output_dir: str, 
                      model_name: str = "Model",
                      save_plot: bool = True,
                      show_plot: bool = True,
                      figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot training loss over epochs from logs and/or checkpoints.
    
    Args:
        output_dir: Training output directory containing logs and checkpoints
        model_name: Name of the model for plot title
        save_plot: Whether to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    """
    
    # Try to get loss data from multiple sources
    epoch_losses = []
    
    # 1. Try to extract from log files
    logs_dir = os.path.join(output_dir, 'logs')
    if os.path.exists(logs_dir):
        for log_file in os.listdir(logs_dir):
            if 'training' in log_file.lower() and log_file.endswith('.log'):
                log_path = os.path.join(logs_dir, log_file)
                losses = extract_loss_from_logs(log_path)
                epoch_losses.extend(losses)
    
    # 2. Try to extract from checkpoints if no log data
    if not epoch_losses:
        losses = extract_loss_from_checkpoints(output_dir)
        epoch_losses.extend(losses)
    
    # 3. Try to extract from final model checkpoint
    if not epoch_losses:
        for model_file in ['vanilla_model.pt', 'symbolic_model.pt', 'model.pt']:
            model_path = os.path.join(output_dir, model_file)
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    training_result = checkpoint.get('training_result', {})
                    epoch_losses_list = training_result.get('epoch_losses', [])
                    
                    if epoch_losses_list:
                        for i, loss in enumerate(epoch_losses_list, 1):
                            epoch_losses.append((i, loss))
                        break
                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
                    continue
    
    if not epoch_losses:
        print(f"No training loss data found in {output_dir}")
        print("Make sure you have either:")
        print("1. Log files in logs/ subdirectory")
        print("2. Checkpoint files with loss information")
        print("3. Final model file with training_result")
        return
    
    # Sort by epoch and remove duplicates
    epoch_losses = sorted(list(set(epoch_losses)))
    
    if not epoch_losses:
        print("No valid loss data found")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = [x[0] for x in epoch_losses]
    losses = [x[1] for x in epoch_losses]
    
    # Plot the line
    ax.plot(epochs, losses, 'o-', linewidth=2, markersize=6, label=f'{model_name} Training Loss')
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'{model_name} Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set integer ticks for epochs if reasonable number
    if max(epochs) <= 20:
        ax.set_xticks(epochs)
    
    # Format y-axis for better readability
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3,3))
    
    # Add value annotations on points if not too many
    if len(epochs) <= 10:
        for epoch, loss in zip(epochs, losses):
            ax.annotate(f'{loss:.4f}', 
                       xy=(epoch, loss), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    if save_plot:
        plot_path = os.path.join(output_dir, f'{model_name.lower()}_training_loss.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

def compare_training_losses(output_dirs: Dict[str, str],
                           save_plot: bool = True,
                           show_plot: bool = True,
                           figsize: Tuple[int, int] = (12, 7)) -> None:
    """
    Compare training losses from multiple models.
    
    Args:
        output_dirs: Dictionary mapping model names to their output directories
        save_plot: Whether to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(output_dirs)))
    
    all_data = {}
    
    for (model_name, output_dir), color in zip(output_dirs.items(), colors):
        # Extract loss data for this model
        epoch_losses = []
        
        # Try logs first
        logs_dir = os.path.join(output_dir, 'logs')
        if os.path.exists(logs_dir):
            for log_file in os.listdir(logs_dir):
                if 'training' in log_file.lower() and log_file.endswith('.log'):
                    log_path = os.path.join(logs_dir, log_file)
                    losses = extract_loss_from_logs(log_path)
                    epoch_losses.extend(losses)
        
        # Try checkpoints if no log data
        if not epoch_losses:
            losses = extract_loss_from_checkpoints(output_dir)
            epoch_losses.extend(losses)
        
        # Try final model
        if not epoch_losses:
            for model_file in ['vanilla_model.pt', 'symbolic_model.pt', 'model.pt']:
                model_path = os.path.join(output_dir, model_file)
                if os.path.exists(model_path):
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        training_result = checkpoint.get('training_result', {})
                        epoch_losses_list = training_result.get('epoch_losses', [])
                        
                        if epoch_losses_list:
                            for i, loss in enumerate(epoch_losses_list, 1):
                                epoch_losses.append((i, loss))
                            break
                    except Exception as e:
                        continue
        
        if epoch_losses:
            epoch_losses = sorted(list(set(epoch_losses)))
            epochs = [x[0] for x in epoch_losses]
            losses = [x[1] for x in epoch_losses]
            
            ax.plot(epochs, losses, 'o-', linewidth=2, markersize=6, 
                   label=model_name, color=color)
            
            all_data[model_name] = (epochs, losses)
            print(f"Loaded {len(epochs)} epochs for {model_name}")
        else:
            print(f"No data found for {model_name} in {output_dir}")
    
    if not all_data:
        print("No training data found for any model")
        return
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3,3))
    
    plt.tight_layout()
    
    # Save the plot
    if save_plot:
        plot_path = 'training_loss_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path}")
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close()
