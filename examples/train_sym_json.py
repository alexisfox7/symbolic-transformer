#!/usr/bin/env python
# examples/train_symbolic_with_json_logging.py
"""
Simplified training script with clean JSON logging integration.
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_preset_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from utils.json_logger2 import create_json_logger
from trainers.json_trainer2 import TrainerWithJSONLogging

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Symbolic Transformer with JSON Logging')
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--max_samples", type=int, default=10000)
    
    # Model
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument("--use_symbolic_ffn", action='store_true', default=True)
    parser.add_argument("--use_v", action='store_true', default=False)
    parser.add_argument("--use_proj", action='store_true', default=False)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--trainer_type', type=str, default='accelerate',
                       choices=['simple', 'accelerate'])
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="outputs/symbolic")
    parser.add_argument("--experiment_name", type=str, default="symbolic_training")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    
    # Misc
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--tokenizer_type", type=str, default="character")
    
    return parser.parse_args()


def setup_logging(output_dir):
    """Setup Python logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*60)
    logger.info("SYMBOLIC TRANSFORMER TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Trainer: {args.trainer_type}")
    logger.info(f"JSON logging: {'Enabled' if not args.disable_json_logging else 'Disabled'}")
    
    # Create configuration
    config = get_preset_config(args.preset)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.use_symbolic_ffn = args.use_symbolic_ffn
    config.use_v = args.use_v
    config.use_proj = args.use_proj
    
    # Initialize tokenizer
    logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer = create_tokenizer(args.tokenizer_type)
    
    # For character tokenizer, build vocab from dataset sample
    if args.tokenizer_type == "character":
        from datasets import load_dataset
        temp_dataset = load_dataset(args.dataset, split=f"train[:{min(args.max_samples, 10000)}]")
        text_field = 'text' if 'text' in temp_dataset.column_names else 'story'
        tokenizer.build_vocab_from_texts(temp_dataset[text_field])
    
    # Update config with tokenizer info
    config.update_from_tokenizer(tokenizer)
    
    # Load data
    logger.info("Loading and preparing data...")
    dataloader, tokenizer = load_and_prepare_data(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=config.block_size,
        batch_size=config.batch_size,
        shuffle=True
    )
    logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")
    
    # Create model
    logger.info("Creating model...")
    model = get_model('symbolic', config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create JSON logger if enabled
    json_logger = None
    if not args.disable_json_logging:
        json_logger = create_json_logger(args.output_dir, args.experiment_name)
        logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    # Create base trainer
    base_trainer = get_trainer(
        trainer_type=args.trainer_type,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        output_dir=args.output_dir,
        log_interval=10  # Console log interval
    )
    
    # Wrap with JSON logging
    trainer = TrainerWithJSONLogging(
        base_trainer, 
        json_logger=json_logger,
        log_interval=args.log_interval
    )
    
    # Train
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    result = trainer.train()
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'training_result': result
    }, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()