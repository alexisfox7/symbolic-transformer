import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_training_and_validation_data(json_files, json_log_steps=50):
    """
    Extract both training and validation data from JSON logs.
    Handles different logging frequencies and aligns them properly.
    """
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Training batch logging frequency: every {json_log_steps} steps")
    
    # Training data (logged every json_log_steps)
    train_batches = []
    train_losses = []
    
    # Validation data (logged less frequently, typically per epoch or every N batches)
    val_batches = []
    val_losses = []
    
    # Epoch boundaries
    epoch_boundaries = []
    
    # Counters for consistent batch numbering
    train_batch_counter = 0
    
    for json_file in sorted(json_files):
        print(f'Processing: {json_file}')
        with open(json_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    
                    # Training batch events
                    if event.get('event_type') == 'batch':
                        loss = event.get('metrics', {}).get('loss')
                        if loss is not None:
                            train_batch_counter += json_log_steps
                            train_batches.append(train_batch_counter)
                            train_losses.append(loss)
                    
                    # Validation events
                    elif event.get('event_type') == 'validation':
                        val_loss = event.get('metrics', {}).get('val_loss') or event.get('metrics', {}).get('loss')
                        global_batch = event.get('metrics', {}).get('global_batch', 0)
                        
                        # Try to get batch info from different possible fields
                        if global_batch == 0:
                            global_batch = event.get('global_batch', 0)
                        if global_batch == 0:
                            global_batch = event.get('step', 0)
                        
                        if val_loss is not None and global_batch > 0:
                            val_batches.append(global_batch)
                            val_losses.append(val_loss)
                    
                    # Epoch end events (for boundaries)
                    elif event.get('event_type') == 'epoch_end':
                        epoch = event.get('epoch', 0)
                        if train_batch_counter > 0:
                            epoch_boundaries.append((train_batch_counter, epoch))
                            
                        # Sometimes validation is logged at epoch end
                        val_loss = event.get('metrics', {}).get('val_loss')
                        if val_loss is not None:
                            val_batches.append(train_batch_counter)
                            val_losses.append(val_loss)
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")
                    continue
    
    print(f"Extracted {len(train_batches)} training points")
    print(f"Extracted {len(val_batches)} validation points")
    
    return {
        'train_batches': train_batches,
        'train_losses': train_losses,
        'val_batches': val_batches,
        'val_losses': val_losses,
        'epoch_boundaries': epoch_boundaries
    }

def interpolate_validation_for_alignment(train_batches, val_batches, val_losses):
    """
    Interpolate validation loss at training batch points for better alignment.
    Only interpolates within the range of validation data.
    """
    if len(val_batches) < 2:
        return [], []
    
    # Sort validation data by batch number
    sorted_pairs = sorted(zip(val_batches, val_losses))
    val_batches_sorted, val_losses_sorted = zip(*sorted_pairs)
    
    # Only interpolate for training batches within validation range
    min_val_batch = min(val_batches_sorted)
    max_val_batch = max(val_batches_sorted)
    
    aligned_batches = []
    aligned_val_losses = []
    
    for batch in train_batches:
        if min_val_batch <= batch <= max_val_batch:
            # Interpolate validation loss at this batch
            interpolated_loss = np.interp(batch, val_batches_sorted, val_losses_sorted)
            aligned_batches.append(batch)
            aligned_val_losses.append(interpolated_loss)
    
    return aligned_batches, aligned_val_losses

def create_comparison_plot(data, output_path, model_name="Symbolic"):
    """Create training vs validation loss comparison plot."""
    
    train_batches = data['train_batches']
    train_losses = data['train_losses']
    val_batches = data['val_batches'] 
    val_losses = data['val_losses']
    epoch_boundaries = data['epoch_boundaries']
    
    if not train_batches:
        print("No training data found!")
        return
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Training and Validation Loss (raw points)
    plt.subplot(2, 2, 1)
    plt.plot(train_batches, train_losses, alpha=0.6, linewidth=1, 
             label='Training Loss', color='orange')
    
    if val_batches and val_losses:
        plt.scatter(val_batches, val_losses, color='red', s=20, alpha=0.8,
                   label='Validation Loss', zorder=5)
    
    plt.title(f'{model_name} Transformer: Training vs Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add epoch boundaries
    for batch, epoch in epoch_boundaries:
        plt.axvline(x=batch, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        plt.text(batch, max(train_losses) * 0.95, f'E{epoch}', 
                rotation=90, verticalalignment='bottom', fontsize=8)
    
    # Plot 2: Smoothed Training Loss
    plt.subplot(2, 2, 2)
    if len(train_losses) > 50:
        window = min(50, len(train_losses) // 5)
        smoothed = []
        for i in range(len(train_losses)):
            start = max(0, i - window // 2)
            end = min(len(train_losses), i + window // 2)
            smoothed.append(sum(train_losses[start:end]) / (end - start))
        plt.plot(train_batches, smoothed, linewidth=2, color='darkorange', 
                label='Smoothed Training Loss')
    else:
        plt.plot(train_batches, train_losses, linewidth=2, color='orange',
                label='Training Loss')
    
    plt.title(f'Smoothed Training Loss ({model_name})')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Aligned Training vs Validation (interpolated)
    plt.subplot(2, 2, 3)
    plt.plot(train_batches, train_losses, alpha=0.4, linewidth=1, 
             label='Training Loss', color='orange')
    
    if val_batches and val_losses:
        # Get interpolated validation for alignment
        aligned_batches, aligned_val_losses = interpolate_validation_for_alignment(
            train_batches, val_batches, val_losses)
        
        if aligned_batches:
            plt.plot(aligned_batches, aligned_val_losses, linewidth=2, 
                    color='red', label='Validation Loss (Interpolated)')
            
            # Also show original validation points
            plt.scatter(val_batches, val_losses, color='darkred', s=15, alpha=0.8,
                       label='Validation Points', zorder=5)
    
    plt.title(f'Training vs Validation Loss (Aligned) - {model_name}')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Loss Gap Analysis
    plt.subplot(2, 2, 4)
    if val_batches and val_losses and len(val_batches) >= 2:
        aligned_batches, aligned_val_losses = interpolate_validation_for_alignment(
            train_batches, val_batches, val_losses)
        
        if aligned_batches:
            # Find corresponding training losses for aligned points
            train_losses_aligned = []
            for batch in aligned_batches:
                # Find closest training batch
                idx = min(range(len(train_batches)), 
                         key=lambda i: abs(train_batches[i] - batch))
                train_losses_aligned.append(train_losses[idx])
            
            # Calculate gap (validation - training)
            loss_gap = [v - t for v, t in zip(aligned_val_losses, train_losses_aligned)]
            
            plt.plot(aligned_batches, loss_gap, linewidth=2, color='purple',
                    label='Val - Train Loss Gap')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Validation - Training Loss Gap - {model_name}')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss Difference')
            plt.grid(True, alpha=0.3)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Comparison plot saved to: {output_path}')
    plt.close()

def print_summary(data):
    """Print summary statistics."""
    train_batches = data['train_batches']
    train_losses = data['train_losses'] 
    val_batches = data['val_batches']
    val_losses = data['val_losses']
    epoch_boundaries = data['epoch_boundaries']
    
    print(f'\n=== TRAINING & VALIDATION SUMMARY ===')
    print(f'Total training steps: {max(train_batches) if train_batches else 0}')
    print(f'Training data points: {len(train_losses)}')
    print(f'Validation data points: {len(val_losses)}')
    
    if train_losses:
        print(f'Final training loss: {train_losses[-1]:.4f}')
        print(f'Best training loss: {min(train_losses):.4f}')
    
    if val_losses:
        print(f'Final validation loss: {val_losses[-1]:.4f}')
        print(f'Best validation loss: {min(val_losses):.4f}')
    
    if epoch_boundaries:
        print(f'Epochs completed: {max([e for _, e in epoch_boundaries])}')
        print(f'Training steps per epoch (avg): {(max(train_batches) / max([e for _, e in epoch_boundaries])):.1f}')

def main():
    # Configuration
    log_dir = './outputs/sym_4gpu_final/logs'
    experiment_name = 'symbolic_4gpu_final'
    json_log_steps = 50
    model_name = "Symbolic"
    
    # Find JSON files
    json_files = []
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.endswith('.jsonl') and experiment_name in f:
                json_files.append(os.path.join(log_dir, f))
    
    if not json_files:
        print(f"No JSON files found in {log_dir} with experiment name '{experiment_name}'")
        return
    
    print(f'Found {len(json_files)} JSON log files')
    
    # Extract data
    data = extract_training_and_validation_data(json_files, json_log_steps)
    
    # Create plot
    output_path = f'{log_dir}/../training_validation_comparison_{model_name.lower()}.png'
    create_comparison_plot(data, output_path, model_name)
    
    # Print summary
    print_summary(data)

if __name__ == "__main__":
    main()