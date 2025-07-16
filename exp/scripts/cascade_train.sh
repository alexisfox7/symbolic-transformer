#!/bin/bash
# Token-Factored Transformer (TFT) Training with Hook System

set -e

# CONFIG
DIR="./outputs/cascade_kronecker_reason"
N=110000
N_EMBD=384
PRESET="small"

NUM_GPUS=2

BATCH_SIZE=8
NUM_EPOCHS=8
LEARNING_RATE=0.0006
LOG_INTERVAL=50
JSON_LOG_STEPS=50

echo "Token-Factored Transformer Cascade Training: $NUM_EPOCHS epochs, $((BATCH_SIZE * NUM_GPUS)) batch size"
echo "Architecture: Stream separation (Xt + Xe) without vocabulary constraints"

# check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available"
    exit 1
fi

# add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

# training
accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    --mixed_precision fp16 \
    exp/examples/train_tft.py \
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
    --validate_every 1 \
    --use_proj kronecker --use_v kronecker \
    --cascade
  #  --use_sparsemax
  
echo "TFT Training completed!"