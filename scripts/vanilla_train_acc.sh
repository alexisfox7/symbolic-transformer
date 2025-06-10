#!/bin/bash
# Simplified Progressive Vanilla Transformer Training with 4 GPUs
# Baseline comparison to symbolic transformer with same parameters

set -e  # Exit on any error

# Configuration - matching symbolic script parameters
DIR="./outputs/vanilla_4gpu_final"
N=110000
EXPERIMENT_NAME="vanilla_4gpu_final"

# Model configuration - matching symbolic script
N_EMBD=384
PRESET="small"

# Multi-GPU configuration - matching symbolic script
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Simplified batch configuration (no gradient accumulation, no stages)
BATCH_SIZE=4  # Direct batch size per GPU

# JSON logging configuration
JSON_LOG_STEPS=50

echo "========================================================"
echo "SIMPLIFIED VANILLA TRANSFORMER 4-GPU TRAINING (BASELINE)"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "Number of GPUs: $NUM_GPUS"
echo "Model size: $N_EMBD dimensions"
echo "JSON logging: Every $JSON_LOG_STEPS batches"
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

# Single training run - all 8 epochs (matching symbolic script)
echo ""
echo "========================================================"
echo "TRAINING: All 8 epochs (Vanilla Transformer Baseline)"
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_vanilla_with_json_logging.py \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --num_epochs 8 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --experiment_name $EXPERIMENT_NAME \
    --learning_rate 0.0012 \
    --clip_grad_norm 1.0 \
    --log_interval 50

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"

# Generate training plots from JSON logs - matching symbolic script
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

# Extract training metrics (simple batch tracking)
all_batches = []
all_losses = []
epoch_boundaries = []

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                # Track batches (much simpler now)
                if event.get('event_type') == 'batch':
                    step = event.get('step', 0)
                    loss = event.get('metrics', {}).get('loss')
                    if loss is not None:
                        all_batches.append(step)
                        all_losses.append(loss)
                        
                elif event.get('event_type') == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    global_batch = event.get('metrics', {}).get('global_batch', 0)
                    if global_batch > 0:
                        epoch_boundaries.append((global_batch, epoch))
            except:
                continue

if all_batches and all_losses:
    # Create training plot
    plt.figure(figsize=(15, 10))
    
    # Plot loss vs batches
    plt.subplot(2, 1, 1)
    plt.plot(all_batches, all_losses, alpha=0.7, linewidth=1, label='Training Loss', color='blue')
    plt.title('Simplified 4-GPU Vanilla Transformer Training Loss vs Batches (Baseline)')
    plt.xlabel('Batch')
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
        plt.plot(all_batches, smoothed, linewidth=2, color='darkblue', label='Smoothed Loss')
        plt.title('Smoothed Training Loss vs Batches (Vanilla Baseline)')
    else:
        plt.plot(all_batches, all_losses, linewidth=2, label='Training Loss', color='blue')
        plt.title('Training Loss vs Batches (Vanilla Baseline)')
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add epoch boundaries
    for batch, epoch in epoch_boundaries:
        plt.axvline(x=batch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plot_path = '$DIR/training_progress_vanilla_baseline.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Training plot saved to: {plot_path}')
    plt.close()
    
    # Print summary
    print(f'\\nVanilla Transformer Training Summary (Baseline):')
    print(f'Total batches: {max(all_batches) if all_batches else 0}')
    print(f'Final loss: {all_losses[-1]:.4f}' if all_losses else 'N/A')
    print(f'Best loss: {min(all_losses):.4f}' if all_losses else 'N/A')
    print(f'Epochs completed: {max([e for _, e in epoch_boundaries]) if epoch_boundaries else 0}')
    print(f'Batches per epoch (avg): {(max(all_batches) / max([e for _, e in epoch_boundaries])) if epoch_boundaries else 0:.1f}')
else:
    print('No valid training data found in JSON logs')
"

# Final summary - matching symbolic script structure but for vanilla
echo ""
echo "========================================================"
echo "SIMPLIFIED 4-GPU VANILLA TRANSFORMER TRAINING COMPLETED!"
echo "========================================================"
echo "Key features (Baseline for comparison with Symbolic):"
echo "  ✓ Standard transformer architecture with positional embeddings"
echo "  ✓ No gradient accumulation complexity"
echo "  ✓ Direct batch sizes instead of effective batch calculations"
echo "  ✓ Simple batch-by-batch training and logging"
echo "  ✓ Clean JSON logs with straightforward batch tracking"
echo ""
echo "Training configuration:"
echo "  Single run: 8 epochs, $BATCH_SIZE×4 = $((BATCH_SIZE * 4)) total batch size"
echo "  Model: Vanilla Transformer with $N_EMBD dimensions"
echo "  Architecture: Standard attention + FFN, learned positional embeddings"
echo ""
echo "Output files:"
echo "  Model: $DIR/vanilla_model.pt"
echo "  Logs: $DIR/logs/"
echo "  Plot: $DIR/training_progress_vanilla_baseline.png"
echo ""
echo "Ready for comparison with symbolic transformer results!"
echo "Use this as baseline to evaluate symbolic interpretability benefits."
echo "========================================================"