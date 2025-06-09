#!/bin/bash

# Progressive Vanilla Transformer Training

DIR="./outputs/vanilla_test"
N=100000

echo "Starting Progressive Vanilla Transformer Training"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "================================================"

# 1. Stage 1: Mini-batch 4, effective 8 (1 epoch)
echo "Stage 1: Training epoch 1 with effective batch size 8"
python examples/train_vanilla.py \
  --batch_size 4 --effective_batch_size 8 \
  --num_epochs 1 --max_samples $N --n_embd 384 \
  --output_dir $DIR

if [ $? -ne 0 ]; then
    echo "Stage 1 failed. Exiting."
    exit 1
fi

# 2. Stage 2: Resume, effective 32 (epochs 2-3)
echo "Stage 2: Resuming for epochs 2-3 with effective batch size 32"
python examples/train_vanilla.py \
  --resume_from_checkpoint $DIR"/checkpoint_epoch_1.pt" \
  --batch_size 4 --effective_batch_size 32 \
  --num_epochs 3 --max_samples $N --n_embd 384 \
  --output_dir $DIR

if [ $? -ne 0 ]; then
    echo "Stage 2 failed. Exiting."
    exit 1
fi

# 3. Stage 3: Resume, effective 64 (epochs 4-8)
echo "Stage 3: Resuming for epochs 4-8 with effective batch size 64"
python examples/train_vanilla.py \
  --resume_from_checkpoint $DIR"/checkpoint_epoch_3.pt" \
  --batch_size 4 --effective_batch_size 64 \
  --num_epochs 8 --max_samples $N --n_embd 384 \
  --output_dir $DIR

if [ $? -ne 0 ]; then
    echo "Stage 3 failed. Exiting."
    exit 1
fi

echo "================================================"
echo "Progressive Vanilla Transformer Training Complete!"
echo "Final model saved to: $DIR/vanilla_model.pt"
echo "Compare with symbolic training using same schedule"