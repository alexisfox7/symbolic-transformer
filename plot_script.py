import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Find all JSON log files in outputs directory
log_dir = './outputs/sym_4gpu_final/logs'
json_files = []
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl') and 'symbolic_4gpu_final' in f:
            json_files.append(os.path.join(log_dir, f))

JSON_LOG_STEPS = 50
print(f'Found {len(json_files)} JSON log files')
print(f'Note: Using JSON_LOG_STEPS={JSON_LOG_STEPS} (logs every {JSON_LOG_STEPS} training steps)')

# Extract training metrics (simple batch tracking)
all_batches = []
all_losses = []
epoch_boundaries = []
batch_counter = 0  # Our own continuous counter
JSON_LOG_STEPS = 50  # Change this to match your actual logging frequency if different

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                # Track batches using our own continuous counter
                if event.get('event_type') == 'batch':
                    loss = event.get('metrics', {}).get('loss')
                    if loss is not None:
                        batch_counter += 1
                        # Convert logged batch number to actual training step
                        actual_batch = batch_counter * JSON_LOG_STEPS
                        all_batches.append(actual_batch)
                        all_losses.append(loss)
                        
                elif event.get('event_type') == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    if batch_counter > 0:
                        actual_batch = batch_counter * JSON_LOG_STEPS
                        epoch_boundaries.append((actual_batch, epoch))
            except:
                continue

if all_batches and all_losses:
    # Create training plot
    plt.figure(figsize=(15, 10))
    
    # Plot loss vs batches
    plt.subplot(2, 1, 1)
    plt.plot(all_batches, all_losses, alpha=0.7, linewidth=1, label='Training Loss', color='orange')
    plt.title('Simplified 4-GPU Symbolic Transformer Training Loss vs Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add epoch boundaries
    for batch, epoch in epoch_boundaries:
        plt.axvline(x=batch, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(batch, max(all_losses) * 0.9, f'E{epoch}', rotation=90, verticalalignment='bottom')
    
    # Plot smoothed loss
    plt.subplot(2, 1, 2)
    if len(all_losses) > 50:
        # Simple moving average
        window = min(50, len(all_losses) // 5)
        smoothed = []
        for i in range(len(all_losses)):
            start = max(0, i - window // 2)
            end = min(len(all_losses), i + window // 2)
            smoothed.append(sum(all_losses[start:end]) / (end - start))
        plt.plot(all_batches, smoothed, linewidth=2, color='darkorange', label='Smoothed Loss')
        plt.title('Smoothed Training Loss vs Training Steps (Symbolic Model)')
    else:
        plt.plot(all_batches, all_losses, linewidth=2, label='Training Loss', color='orange')
        plt.title('Training Loss vs Training Steps (Symbolic Model)')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add epoch boundaries
    for batch, epoch in epoch_boundaries:
        plt.axvline(x=batch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plot_path = './outputs/sym_4gpu_final/training_progress_symbolic.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Training plot saved to: {plot_path}')
    plt.close()
    
    # Print summary
    print(f'\nSymbolic Transformer Training Summary:')
    print(f'Total training steps: {max(all_batches) if all_batches else 0}')
    print(f'Final loss: {all_losses[-1]:.4f}' if all_losses else 'N/A')
    print(f'Best loss: {min(all_losses):.4f}' if all_losses else 'N/A')
    print(f'Epochs completed: {max([e for _, e in epoch_boundaries]) if epoch_boundaries else 0}')
    print(f'Training steps per epoch (avg): {(max(all_batches) / max([e for _, e in epoch_boundaries])) if epoch_boundaries else 0:.1f}')
else:
    print('No valid training data found in JSON logs')