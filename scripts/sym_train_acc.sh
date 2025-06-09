#!/bin/bash
# Progressive Symbolic Transformer Training with 4 GPUs and JSON Logging
# Based on sym-train-2.sh but optimized for multi-GPU training

set -e  # Exit on any error

# Configuration
DIR="./outputs/sym_4gpu_json"
N=100000
EXPERIMENT_NAME="symbolic_4gpu_progressive"

# Model configuration
N_EMBD=384
PRESET="small"

# Multi-GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# JSON logging configuration
JSON_LOG_STEPS=128  # Log more frequently with 4 GPUs (more steps per second)

echo "========================================================"
echo "SYMBOLIC TRANSFORMER 4-GPU PROGRESSIVE TRAINING"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "Number of GPUs: $NUM_GPUS"
echo "Model size: $N_EMBD dimensions"
echo "JSON logging: Every $JSON_LOG_STEPS steps"
echo "Experiment name: $EXPERIMENT_NAME"
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

# Stage 1: Initial training (epoch 1)
# Start with small effective batch size for stability
echo ""
echo "========================================================"
echo "STAGE 1: Initial training (epoch 1)"
echo "Mini-batch: 4, Effective batch: 32 (4 GPUs × 4 batch × 2 accum)"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_symbolic_with_json_logging.py \
    --use_proj --use_v \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_epochs 1 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --experiment_name "${EXPERIMENT_NAME}_stage1" \
    --learning_rate 3e-4 \
    --clip_grad_norm 1.0 \
    2>/dev/null

if [ $? -ne 0 ]; then
    echo "Stage 1 failed. Exiting."
    exit 1
fi

echo "Stage 1 completed successfully!"

# Stage 2: Scale up effective batch size (epochs 2-3)
# Increase effective batch size for better convergence
echo ""
echo "========================================================"
echo "STAGE 2: Scaled training (epochs 2-3)"
echo "Mini-batch: 8, Effective batch: 128 (4 GPUs × 8 batch × 4 accum)"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_symbolic_with_json_logging.py \
    --use_proj --use_v \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --resume_from_checkpoint "$DIR/checkpoint_epoch_1.pt" \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --experiment_name "${EXPERIMENT_NAME}_stage2" \
    --learning_rate 2e-4 \
    --clip_grad_norm 1.0 \
    2>/dev/null

if [ $? -ne 0 ]; then
    echo "Stage 2 failed. Exiting."
    exit 1
fi

echo "Stage 2 completed successfully!"

# Stage 3: Large batch training (epochs 4-8)
# Maximum effective batch size for final training
echo ""
echo "========================================================"
echo "STAGE 3: Large batch training (epochs 4-8)"
echo "Mini-batch: 8, Effective batch: 256 (4 GPUs × 8 batch × 8 accum)"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_symbolic_with_json_logging.py \
    --use_proj --use_v \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --resume_from_checkpoint "$DIR/checkpoint_epoch_3.pt" \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_epochs 8 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --experiment_name "${EXPERIMENT_NAME}_stage3" \
    --learning_rate 1e-4 \
    --clip_grad_norm 1.0 \
    2>/dev/null

if [ $? -ne 0 ]; then
    echo "Stage 3 failed. Exiting."
    exit 1
fi

echo "Stage 3 completed successfully!"

# Generate training plots from JSON logs
echo ""
echo "========================================================"
echo "GENERATING TRAINING PLOTS FROM JSON LOGS"
echo "========================================================"

python -c "
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Find all JSON log files
log_dir = '$DIR/logs'
json_files = []
if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl') and '$EXPERIMENT_NAME' in f:
            json_files.append(os.path.join(log_dir, f))

print(f'Found {len(json_files)} JSON log files')

# Extract training metrics
all_steps = []
all_losses = []
epoch_boundaries = []

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                if event.get('event_type') == 'training_step':
                    step = event.get('step', 0)
                    loss = event.get('metrics', {}).get('loss')
                    if loss is not None:
                        all_steps.append(step)
                        all_losses.append(loss)
                elif event.get('event_type') == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    step = event.get('metrics', {}).get('global_step', 0)
                    if step > 0:
                        epoch_boundaries.append((step, epoch))
            except:
                continue

if all_steps and all_losses:
    # Create training plot
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(all_steps, all_losses, alpha=0.7, linewidth=1)
    plt.title('4-GPU Symbolic Transformer Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Add epoch boundaries
    for step, epoch in epoch_boundaries:
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(step, max(all_losses), f'E{epoch}', rotation=90, verticalalignment='top')
    
    # Plot smoothed loss
    plt.subplot(2, 1, 2)
    if len(all_losses) > 100:
        # Simple moving average
        window = min(100, len(all_losses) // 10)
        smoothed = []
        for i in range(len(all_losses)):
            start = max(0, i - window // 2)
            end = min(len(all_losses), i + window // 2)
            smoothed.append(sum(all_losses[start:end]) / (end - start))
        plt.plot(all_steps, smoothed, linewidth=2, color='orange')
        plt.title('Smoothed Training Loss')
    else:
        plt.plot(all_steps, all_losses, linewidth=2)
        plt.title('Training Loss')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Add epoch boundaries
    for step, epoch in epoch_boundaries:
        plt.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plot_path = '$DIR/training_progress.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Training plot saved to: {plot_path}')
    plt.close()
    
    # Print summary
    print(f'\\nTraining Summary:')
    print(f'Total steps: {max(all_steps) if all_steps else 0}')
    print(f'Final loss: {all_losses[-1]:.4f}' if all_losses else 'N/A')
    print(f'Best loss: {min(all_losses):.4f}' if all_losses else 'N/A')
    print(f'Epochs completed: {max([e for _, e in epoch_boundaries]) if epoch_boundaries else 0}')
else:
    print('No training data found in JSON logs')
"

# Final summary
echo ""
echo "========================================================"
echo "4-GPU PROGRESSIVE TRAINING COMPLETED!"
echo "========================================================"
echo "Training stages:"
echo "  Stage 1: Epoch 1, Effective batch 32"
echo "  Stage 2: Epochs 2-3, Effective batch 128" 
echo "  Stage 3: Epochs 4-8, Effective batch 256"
echo ""
echo "Output files:"
echo "  Model: $DIR/symbolic_model.pt"
echo "  Logs: $DIR/logs/"
echo "  Plot: $DIR/training_progress.png"
echo ""
echo "JSON logs contain detailed step-by-step metrics"
echo "Use the JSON logs for detailed analysis and visualization"
echo "========================================================"