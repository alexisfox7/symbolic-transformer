#!/bin/bash
#  Vanilla Transformer Training with 4 GPUs using Hook System

set -e  # exit on any error

# CONFIG -----
DIR="./outputs/vanilla_4gpu_modern"
N=110000
EXPERIMENT_NAME="vanilla_4gpu_modern"

# model   
N_EMBD=384
PRESET="small"

# multi-gpu 
#NOTE this is not flexible
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# training
BATCH_SIZE=4  #REVIEW should be per GPU
NUM_EPOCHS=8
LEARNING_RATE=0.0012

# logging 
LOG_INTERVAL=50
JSON_LOG_STEPS=50

echo "========================================================"
echo "MODERN VANILLA TRANSFORMER 4-GPU TRAINING"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "GPUs: $NUM_GPUS"
echo "Model: $N_EMBD dimensions, $PRESET preset"
echo "Batch size: $BATCH_SIZE per GPU (total: $((BATCH_SIZE * NUM_GPUS)))"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "========================================================"

# create output directory
mkdir -p $DIR

# check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available or Python/PyTorch not working"
    exit 1
fi

# accelerate config 
if [ ! -f "./src/config/accelerate_config_4gpu.yaml" ]; then
    echo "ERROR: Accelerate config not found at ./src/config/accelerate_config_4gpu.yaml"
    echo "Creating basic config..."
    mkdir -p ./src/config
    cat > ./src/config/accelerate_config_4gpu.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: '0,1,2,3'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# training
echo ""
echo "========================================================"
echo "STARTING TRAINING"
echo "========================================================"

accelerate launch \
    --config_file ./src/config/accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_vanilla_modern.py \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --log_interval $LOG_INTERVAL \
    --json_log_steps $JSON_LOG_STEPS \
    --clip_grad_norm 1.0 \
    --val_ratio 0.1 \
    --validate_every 2

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"

# generate training plots from JSON logs
echo ""
echo "========================================================"
echo "GENERATING TRAINING PLOTS FROM JSON LOGS"
echo "========================================================"

python3 -c "
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Find JSON log files
log_dir = '$DIR/logs'
json_files = []
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl'):
            json_files.append(os.path.join(log_dir, f))

print(f'Found {len(json_files)} JSON log files')

# Extract metrics from hook-based JSON logs
all_steps = []
all_losses = []
epoch_losses = []
epochs = []
val_losses = []
val_epochs = []

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                event_type = event.get('event')
                
                # Hook-based batch events
                if event_type == 'batch':
                    step = len(all_steps)  # Simple step counter
                    loss = event.get('loss')
                    if loss is not None:
                        all_steps.append(step)
                        all_losses.append(loss)
                
                # Hook-based epoch events        
                elif event_type == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    loss = event.get('loss')
                    if loss is not None:
                        epochs.append(epoch)
                        epoch_losses.append(loss)
                
                # Validation events (if using validation hook)
                elif 'validation' in event.get('event', ''):
                    epoch = event.get('epoch', 0) 
                    val_loss = event.get('loss')
                    if val_loss is not None:
                        val_epochs.append(epoch)
                        val_losses.append(val_loss)
                        
            except json.JSONDecodeError:
                continue

if all_steps and all_losses:
    # Create training plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Batch-level loss
    plt.subplot(2, 2, 1)
    plt.plot(all_steps, all_losses, alpha=0.7, linewidth=1, color='blue', label='Training Loss')
    plt.title('Vanilla Transformer: Batch-Level Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Smoothed loss
    plt.subplot(2, 2, 2)
    if len(all_losses) > 50:
        window = min(50, len(all_losses) // 10)
        smoothed = np.convolve(all_losses, np.ones(window)/window, mode='valid')
        smoothed_steps = all_steps[window-1:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, color='darkblue', label=f'Smoothed Loss (window={window})')
    else:
        plt.plot(all_steps, all_losses, linewidth=2, color='blue', label='Training Loss')
    plt.title('Smoothed Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Epoch-level metrics
    plt.subplot(2, 2, 3)
    if epochs and epoch_losses:
        plt.plot(epochs, epoch_losses, 'o-', linewidth=2, markersize=6, color='green', label='Epoch Avg Loss')
    if val_epochs and val_losses:
        plt.plot(val_epochs, val_losses, 's-', linewidth=2, markersize=6, color='red', label='Validation Loss')
    plt.title('Epoch-Level Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Loss distribution
    plt.subplot(2, 2, 4)
    plt.hist(all_losses, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Training Loss Distribution')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = '$DIR/training_progress_modern.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Training plot saved to: {plot_path}')
    plt.close()
    
    # Print summary
    print(f'\\nModern Vanilla Transformer Training Summary:')
    print(f'Total training steps: {len(all_steps)}')
    print(f'Final loss: {all_losses[-1]:.4f}' if all_losses else 'N/A')
    print(f'Best loss: {min(all_losses):.4f}' if all_losses else 'N/A')
    print(f'Epochs completed: {max(epochs) if epochs else 0}')
    
    if val_losses:
        print(f'Final validation loss: {val_losses[-1]:.4f}')
        print(f'Best validation loss: {min(val_losses):.4f}')
    
else:
    print('No valid training data found in JSON logs')
    print('Check that the training script is using the hook system correctly')
"

# Final summary
echo ""
echo "========================================================"
echo "MODERN VANILLA TRANSFORMER TRAINING COMPLETED!"
echo "========================================================"
echo "Key improvements over old script:"
echo "  ✓ Uses modern hook system instead of old JSON logging"
echo "  ✓ Simplified configuration and cleaner output"
echo "  ✓ Better error handling and validation"
echo "  ✓ Modern plotting with multiple views"
echo ""
echo "Training configuration:"
echo "  Model: Vanilla Transformer, $N_EMBD dimensions"
echo "  Training: $NUM_EPOCHS epochs, $((BATCH_SIZE * NUM_GPUS)) total batch size"
echo "  Data: $N samples from TinyStories"
echo ""
echo "Output files:"
echo "  Model: $DIR/vanilla_model.pt"
echo "  Logs: $DIR/logs/"
echo "  Plot: $DIR/training_progress_modern.png"
echo ""
echo "Ready for comparison with symbolic transformer!"
echo "========================================================"