import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def extract_training_data(json_files, json_log_steps=50):
    """Extract training data from JSON logs."""
    train_batches = []
    train_losses = []
    epoch_boundaries = []
    train_batch_counter = 0
    
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    
                    if event.get('event_type') == 'batch':
                        loss = event.get('metrics', {}).get('loss')
                        if loss is not None:
                            train_batch_counter += json_log_steps
                            train_batches.append(train_batch_counter)
                            train_losses.append(loss)
                    
                    elif event.get('event_type') == 'epoch_end':
                        epoch = event.get('epoch', 0)
                        if train_batch_counter > 0:
                            epoch_boundaries.append((train_batch_counter, epoch))
                            
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    
    return train_batches, train_losses, epoch_boundaries

def extract_validation_data(batch_metrics_dir):
    """Extract validation data from batch_metrics directory."""
    val_batches = []
    val_losses = []
    
    if not os.path.exists(batch_metrics_dir):
        return val_batches, val_losses
    
    # Look for all jsonl files in batch_metrics
    for filename in os.listdir(batch_metrics_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(batch_metrics_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event.get('event_type') == 'validation':
                                batch = event.get('global_batch', 0) or event.get('step', 0) or event.get('metrics', {}).get('global_batch', 0)
                                loss = event.get('metrics', {}).get('val_loss') or event.get('metrics', {}).get('loss')
                                if batch > 0 and loss is not None:
                                    val_batches.append(batch)
                                    val_losses.append(loss)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
    
    return val_batches, val_losses

def create_comparison_plot(vanilla_data, symbolic_data, output_path):
    """Create comparison plot with training and validation for both models."""
    
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Training Loss Comparison
    plt.subplot(2, 2, 1)
    
    # Vanilla training
    if vanilla_data['train_batches']:
        plt.plot(vanilla_data['train_batches'], vanilla_data['train_losses'], 
                alpha=0.7, linewidth=1.5, label='Vanilla Training', color='blue')
    
    # Symbolic training  
    if symbolic_data['train_batches']:
        plt.plot(symbolic_data['train_batches'], symbolic_data['train_losses'],
                alpha=0.7, linewidth=1.5, label='Symbolic Training', color='orange')
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Validation Loss Comparison
    plt.subplot(2, 2, 2)
    
    # Vanilla validation (dashed)
    if vanilla_data['val_batches']:
        plt.plot(vanilla_data['val_batches'], vanilla_data['val_losses'],
                linestyle='--', linewidth=2, label='Vanilla Validation', color='blue', alpha=0.8)
    
    # Symbolic validation (dashed)
    if symbolic_data['val_batches']:
        plt.plot(symbolic_data['val_batches'], symbolic_data['val_losses'],
                linestyle='--', linewidth=2, label='Symbolic Validation', color='orange', alpha=0.8)
    
    plt.title('Validation Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Combined Training + Validation
    plt.subplot(2, 2, 3)
    
    # Vanilla
    if vanilla_data['train_batches']:
        plt.plot(vanilla_data['train_batches'], vanilla_data['train_losses'],
                alpha=0.6, linewidth=1, label='Vanilla Training', color='blue')
    if vanilla_data['val_batches']:
        plt.plot(vanilla_data['val_batches'], vanilla_data['val_losses'],
                linestyle='--', linewidth=2, label='Vanilla Validation', color='blue')
    
    # Symbolic
    if symbolic_data['train_batches']:
        plt.plot(symbolic_data['train_batches'], symbolic_data['train_losses'],
                alpha=0.6, linewidth=1, label='Symbolic Training', color='orange')
    if symbolic_data['val_batches']:
        plt.plot(symbolic_data['val_batches'], symbolic_data['val_losses'],
                linestyle='--', linewidth=2, label='Symbolic Validation', color='orange')
    
    plt.title('Training vs Validation: Vanilla vs Symbolic')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Smoothed Training Comparison
    plt.subplot(2, 2, 4)
    
    # Vanilla smoothed
    if vanilla_data['train_batches'] and len(vanilla_data['train_losses']) > 20:
        window = min(50, len(vanilla_data['train_losses']) // 5)
        smoothed = []
        for i in range(len(vanilla_data['train_losses'])):
            start = max(0, i - window // 2)
            end = min(len(vanilla_data['train_losses']), i + window // 2)
            smoothed.append(sum(vanilla_data['train_losses'][start:end]) / (end - start))
        plt.plot(vanilla_data['train_batches'], smoothed, 
                linewidth=2, color='darkblue', label='Vanilla (Smoothed)')
    
    # Symbolic smoothed
    if symbolic_data['train_batches'] and len(symbolic_data['train_losses']) > 20:
        window = min(50, len(symbolic_data['train_losses']) // 5)
        smoothed = []
        for i in range(len(symbolic_data['train_losses'])):
            start = max(0, i - window // 2)
            end = min(len(symbolic_data['train_losses']), i + window // 2)
            smoothed.append(sum(symbolic_data['train_losses'][start:end]) / (end - start))
        plt.plot(symbolic_data['train_batches'], smoothed,
                linewidth=2, color='darkorange', label='Symbolic (Smoothed)')
    
    plt.title('Smoothed Training Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Comparison plot saved to: {output_path}')
    plt.close()

def print_summary(vanilla_data, symbolic_data):
    """Print summary statistics for both models."""
    print('\n=== MODEL COMPARISON SUMMARY ===')
    
    print('\nVANILLA TRANSFORMER:')
    if vanilla_data['train_batches']:
        print(f'  Training steps: {max(vanilla_data["train_batches"])}')
        print(f'  Final train loss: {vanilla_data["train_losses"][-1]:.4f}')
        print(f'  Best train loss: {min(vanilla_data["train_losses"]):.4f}')
    if vanilla_data['val_batches']:
        print(f'  Final val loss: {vanilla_data["val_losses"][-1]:.4f}')
        print(f'  Best val loss: {min(vanilla_data["val_losses"]):.4f}')
    
    print('\nSYMBOLIC TRANSFORMER:')
    if symbolic_data['train_batches']:
        print(f'  Training steps: {max(symbolic_data["train_batches"])}')
        print(f'  Final train loss: {symbolic_data["train_losses"][-1]:.4f}')
        print(f'  Best train loss: {min(symbolic_data["train_losses"]):.4f}')
    if symbolic_data['val_batches']:
        print(f'  Final val loss: {symbolic_data["val_losses"][-1]:.4f}')
        print(f'  Best val loss: {min(symbolic_data["val_losses"]):.4f}')

def main():
    # Paths for both models
    vanilla_logs = './outputs/vanilla_4gpu_final/logs'
    vanilla_batch_metrics = './outputs/vanilla_4gpu_final/batch_metrics'
    
    symbolic_logs = './outputs/sym_4gpu_final/logs'
    symbolic_batch_metrics = './outputs/sym_4gpu_final/batch_metrics'
    
    json_log_steps = 50
    
    # Extract vanilla data
    vanilla_json_files = []
    if os.path.exists(vanilla_logs):
        for f in os.listdir(vanilla_logs):
            if f.endswith('.jsonl') and 'vanilla' in f:
                vanilla_json_files.append(os.path.join(vanilla_logs, f))
    
    vanilla_train_batches, vanilla_train_losses, vanilla_epochs = extract_training_data(vanilla_json_files, json_log_steps)
    vanilla_val_batches, vanilla_val_losses = extract_validation_data(vanilla_batch_metrics)
    
    vanilla_data = {
        'train_batches': vanilla_train_batches,
        'train_losses': vanilla_train_losses,
        'val_batches': vanilla_val_batches,
        'val_losses': vanilla_val_losses,
        'epoch_boundaries': vanilla_epochs
    }
    
    # Extract symbolic data
    symbolic_json_files = []
    if os.path.exists(symbolic_logs):
        for f in os.listdir(symbolic_logs):
            if f.endswith('.jsonl') and 'symbolic' in f:
                symbolic_json_files.append(os.path.join(symbolic_logs, f))
    
    symbolic_train_batches, symbolic_train_losses, symbolic_epochs = extract_training_data(symbolic_json_files, json_log_steps)
    symbolic_val_batches, symbolic_val_losses = extract_validation_data(symbolic_batch_metrics)
    
    symbolic_data = {
        'train_batches': symbolic_train_batches,
        'train_losses': symbolic_train_losses,
        'val_batches': symbolic_val_batches,
        'val_losses': symbolic_val_losses,
        'epoch_boundaries': symbolic_epochs
    }
    
    print(f'Vanilla: {len(vanilla_train_batches)} train, {len(vanilla_val_batches)} val points')
    print(f'Symbolic: {len(symbolic_train_batches)} train, {len(symbolic_val_batches)} val points')
    
    # Create comparison plot
    output_path = './outputs/vanilla_vs_symbolic_comparison.png'
    create_comparison_plot(vanilla_data, symbolic_data, output_path)
    
    # Print summary
    print_summary(vanilla_data, symbolic_data)

if __name__ == "__main__":
    main()