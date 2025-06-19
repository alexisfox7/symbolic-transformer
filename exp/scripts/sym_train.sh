#!/bin/bash
# Symbolic Transformer Training with Hook System

set -e

# CONFIG
DIR="./outputs/sym_4gpu"
N=110000
N_EMBD=384
PRESET="small"

NUM_GPUS=4

BATCH_SIZE=4
NUM_EPOCHS=8
LEARNING_RATE=0.0012
LOG_INTERVAL=50
JSON_LOG_STEPS=50

echo "Symbolic Transformer Training: $NUM_EPOCHS epochs, $((BATCH_SIZE * NUM_GPUS)) batch size"

# i dont think i need to make output directory

# check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available"
    exit 1
fi

# training
accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    --mixed_precision fp16 \
    exp/examples/train_symbolic.py \
    --use_proj --use_v \
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

echo "Training completed!"