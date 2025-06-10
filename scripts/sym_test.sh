#!/bin/bash
# Symbolic Transformer Training with Simplified JSON Logging
# Matches sym_train_acc.sh structure but with clean JSON logging

set -e  # Exit on any error

# Configuration - matching original script
DIR="./outputs/sym_4gpu_json"
N=110000
EXPERIMENT_NAME="symbolic_4gpu_json"

# Model configuration - matching original
N_EMBD=384
PRESET="small"

# Multi-GPU configuration - matching original
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Batch configuration (no gradient accumulation)
BATCH_SIZE=4  # Direct batch size per GPU

# JSON logging configuration
LOG_INTERVAL=100  # Log every 100 steps (simplified from JSON_LOG_STEPS)

echo "========================================================"
echo "SYMBOLIC TRANSFORMER 4-GPU TRAINING WITH JSON LOGGING"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "Number of GPUs: $NUM_GPUS"
echo "Model size: $N_EMBD dimensions"
echo "JSON logging: Every $LOG_INTERVAL steps"
echo "Experiment name: $EXPERIMENT_NAME"
echo ""
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "========================================================"

# Create output directory
mkdir -p $DIR

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available or Python/PyTorch not working"
    exit 1
fi

# Single training run - all 8 epochs
echo ""
echo "========================================================"
echo "TRAINING: All 8 epochs"
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "========================================================"

# Using the simplified training script
accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_symbolic_simplified.py \
    --use_proj --use_v \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --num_epochs 8 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --log_interval $LOG_INTERVAL \
    --experiment_name $EXPERIMENT_NAME \
    --learning_rate 0.0012 \
    --clip_grad_norm 1.0 \
    --console_log_interval 32

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"

# Generate training plots from JSON logs
echo ""
echo "========================================================"
echo "GENERATING TRAINING PLOTS FROM JSON LOGS"
echo "========================================================"

python -c "
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Find the JSON log file
log_dir = '$DIR/logs'
json_file = None
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl') and '$EXPERIMENT_NAME' in f:
            json_file = os.path.join(log_dir, f)
            break

if not json_file:
    print('ERROR: No JSON log file found')
    exit(1)

print(f'Processing: {json_file}')

# Read all events
events = []
with open(json_file, 'r') as f:
    for line in f:
        if line.strip():
            events.append(json.loads(line))

# Extract metrics
steps = []
losses = []
epoch_boundaries = []
current_epoch = 0

for event in events:
    if event['event'] == 'metrics':
        steps.append(event['step'])
        losses.append(event['metrics']['loss'])
    elif event['event'] == 'epoch_summary':
        epoch = event['epoch']
        if epoch > current_epoch and steps:
            epoch_boundaries.append((steps[-1], epoch))
            current_epoch = epoch

if not steps:
    print('No training metrics found')
    exit(1)

# Create plot
plt.figure(figsize=(12, 6))

# Smooth the losses for cleaner visualization
window_size = min(50, len(losses) // 10)
if window_size > 1:
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = steps[:len(smoothed_losses)]
    plt.plot(smoothed_steps, smoothed_losses, 'b-', linewidth=2, label='Smoothed Loss')
    plt.plot(steps, losses, 'b-', alpha=0.3, linewidth=0.5, label='Raw Loss')
else:
    plt.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')

plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Progress - Symbolic Transformer (4 GPUs)')
plt.grid(True, alpha=0.3)
plt.legend()

# Add epoch boundaries
for step, epoch in epoch_boundaries:
    plt.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(step, plt.ylim()[1], f'Epoch {epoch}', rotation=90, va='top', ha='right')

plt.tight_layout()
plot_path = '$DIR/training_progress.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f'Training plot saved to: {plot_path}')
plt.close()

# Print summary statistics
print(f'\\nTraining Summary:')
print(f'Total steps: {steps[-1]}')
print(f'Final loss: {losses[-1]:.4f}')
print(f'Best loss: {min(losses):.4f}')
print(f'Epochs completed: {current_epoch}')
if epoch_boundaries:
    steps_per_epoch = epoch_boundaries[0][0]
    print(f'Steps per epoch: ~{steps_per_epoch}')

# Create epoch summary plot
if epoch_boundaries:
    plt.figure(figsize=(10, 6))
    
    epoch_nums = []
    epoch_avg_losses = []
    
    start_idx = 0
    for step, epoch in epoch_boundaries:
        step_idx = steps.index(step) if step in steps else len(steps)-1
        epoch_losses = losses[start_idx:step_idx+1]
        if epoch_losses:
            epoch_nums.append(epoch)
            epoch_avg_losses.append(np.mean(epoch_losses))
        start_idx = step_idx + 1
    
    plt.plot(epoch_nums, epoch_avg_losses, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.grid(True, alpha=0.3)
    plt.xticks(epoch_nums)
    
    plt.tight_layout()
    epoch_plot_path = '$DIR/epoch_losses.png'
    plt.savefig(epoch_plot_path, dpi=300, bbox_inches='tight')
    print(f'Epoch plot saved to: {epoch_plot_path}')
    plt.close()
"

# Final summary
echo ""
echo "========================================================"
echo "4-GPU TRAINING WITH JSON LOGGING COMPLETED!"
echo "========================================================"
echo "Training configuration:"
echo "  Model: Symbolic Transformer ($PRESET, $N_EMBD dim)"
echo "  Dataset: $N samples"
echo "  Training: 8 epochs, batch size $BATCH_SIZE×$NUM_GPUS"
echo "  Learning rate: 0.0012"
echo ""
echo "Output files:"
echo "  Model: $DIR/model.pt"
echo "  JSON logs: $DIR/logs/${EXPERIMENT_NAME}_*.jsonl"
echo "  Training plot: $DIR/training_progress.png"
echo "  Epoch plot: $DIR/epoch_losses.png"
echo ""
echo "JSON log format (simplified):"
echo "  - Each line is a JSON event"
echo "  - Events: start, config, metrics, epoch_summary, end"
echo "  - Clean structure for easy parsing"
echo "========================================================"