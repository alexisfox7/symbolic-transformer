#!/usr/bin/env python
# examples/train_symbolic_simplified.py
"""
Simplified symbolic transformer training with clean JSON logging.
Matches the structure of train_symbolic_with_json_logging.py but uses
our simplified logging approach.
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from modelold import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset

# Import our simplified JSON logging
from utils.json_logger2 import create_json_logger
from trainers.json_trainer2 import TrainerWithJSONLogging

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Symbolic Transformer with Simplified JSON Logging')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--block_size', type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", action='store_true')
    
    # Symbolic-specific parameters
    parser.add_argument("--use_symbolic_ffn", action='store_true', default=True)
    parser.add_argument("--no_symbolic_ffn", action='store_false', dest='use_symbolic_ffn')
    parser.add_argument("--use_vocab_refinement", action='store_true', default=False)
    parser.add_argument("--use_v", action='store_true', default=False)
    parser.add_argument("--use_proj", action='store_true', default=False)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--trainer_type', type=str, default='accelerate',
                       choices=['simple', 'accelerate'])
    
    # Logging parameters
    parser.add_argument("--output_dir", type=str, default="outputs/symbolic")
    parser.add_argument("--experiment_name", type=str, default="symbolic_training")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log metrics to JSON every N steps")
    parser.add_argument("--console_log_interval", type=int, default=10,
                       help="Log to console every N batches")
    parser.add_argument("--disable_json_logging", action="store_true")
    
    # Generation parameters
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--generation_max_len", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    
    # Other
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--tokenizer_type", type=str, default="character")
    parser.add_argument("--save_model_filename", type=str, default="model.pt")
    
    return parser.parse_args()


def create_symbolic_config(args):
    """Create configuration for the symbolic transformer."""
    config = get_preset_config(args.preset)
    
    # Override with command line arguments
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    if args.n_head is not None:
        config.n_head = args.n_head
    if args.n_embd is not None:
        config.n_embd = args.n_embd
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.bias is not None:
        config.bias = args.bias
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    # Symbolic-specific parameters
    config.use_symbolic_ffn = args.use_symbolic_ffn
    config.use_vocab_refinement = args.use_vocab_refinement
    config.use_v = args.use_v
    config.use_proj = args.use_proj
    
    # Training parameters
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    config.generation_max_len = args.generation_max_len
    config.temperature = args.temperature
    config.top_k = args.top_k
    
    config.__post_init__()
    return config


def setup_logging(output_dir):
    """Setup basic logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training function."""
    args = parse_args()
    logger = setup_logging(args.output_dir)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*60)
    logger.info("SYMBOLIC TRANSFORMER TRAINING (SIMPLIFIED)")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Trainer: {args.trainer_type}")
    logger.info(f"JSON logging: {'Enabled' if not args.disable_json_logging else 'Disabled'}")
    if not args.disable_json_logging:
        logger.info(f"JSON log interval: every {args.log_interval} steps")
    
    # Create configuration
    config = create_symbolic_config(args)
    
    # Initialize tokenizer
    logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
    if args.tokenizer_type == "character":
        # Build character vocab from dataset sample
        temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
        temp_dataset_args = [args.dataset]
        if args.dataset_config:
            temp_dataset_args.append(args.dataset_config)
        
        temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
        
        # Find text field
        if 'text' in temp_dataset.column_names:
            text_samples = temp_dataset['text']
        elif 'story' in temp_dataset.column_names:
            text_samples = temp_dataset['story']
        else:
            text_field = next((col for col in temp_dataset.column_names 
                             if temp_dataset.features[col].dtype == 'string'), None)
            if not text_field:
                logger.error(f"Could not find text column. Available: {temp_dataset.column_names}")
                sys.exit(1)
            text_samples = temp_dataset[text_field]
        
        tokenizer = create_tokenizer(args.tokenizer_type)
        tokenizer.build_vocab_from_texts([str(item) for item in text_samples])
    else:
        tokenizer = create_tokenizer(args.tokenizer_type)
    
    # Update config with tokenizer info
    config.update_from_tokenizer(tokenizer)
    
    # Print configuration
    print_config(config, dataset_name=args.dataset)
    
    # Setup JSON logging
    json_logger = None
    if not args.disable_json_logging:
        json_logger = create_json_logger(args.output_dir, args.experiment_name)
        
        # Log initial configuration
        json_logger.log_config({
            'model_config': {
                'preset': args.preset,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'vocab_size': config.vocab_size,
                'block_size': config.block_size,
                'use_symbolic_ffn': config.use_symbolic_ffn,
                'use_v': config.use_v,
                'use_proj': config.use_proj,
            },
            'training_config': {
                'dataset': args.dataset,
                'max_samples': args.max_samples,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'trainer_type': args.trainer_type,
            },
            'system_config': {
                'device': str(device),
                'tokenizer_type': args.tokenizer_type,
            }
        })
        logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    dataloader, tokenizer = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=config.block_size,
        batch_size=config.batch_size,
        mlm=False,
        split='train',
        shuffle=True
    )
    logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")
    
    # Create model
    logger.info("Creating model...")
    model = get_model('Symbolic', config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create base trainer
    base_trainer = get_trainer(
        trainer_type=args.trainer_type,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        output_dir=args.output_dir,
        clip_grad_norm=args.clip_grad_norm,
        log_interval=args.console_log_interval
    )
    
    # Wrap with JSON logging if enabled
    if json_logger:
        trainer = TrainerWithJSONLogging(
            base_trainer,
            json_logger=json_logger,
            log_interval=args.log_interval
        )
    else:
        trainer = base_trainer
    
    # Train the model
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    training_result = trainer.train()
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    
    # Save final model
    model_path = os.path.join(args.output_dir, args.save_model_filename)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.num_epochs,
        'config': config,
        'tokenizer': tokenizer,
        'training_args': vars(args),
        'training_result': training_result,
        'timestamp': datetime.now().isoformat(),
    }
    
    torch.save(save_dict, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test generation
    if not args.skip_generation:
        logger.info("="*60)
        logger.info("TESTING GENERATION")
        logger.info("="*60)
        
        test_prompts = [
            "The brave knight",
            "Once upon a time",
            "Spotty the dog loved",
            "In the magical forest",
        ]
        
        examples = []
        for prompt in test_prompts:
            generated = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=config.generation_max_len,
                temperature=config.temperature,
                top_k=config.top_k,
                device=device
            )
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-" * 40)
            
            examples.append({
                "prompt": prompt,
                "generated": generated
            })
        
        # Log generation examples to JSON
        if json_logger:
            json_logger.log_generation(config.num_epochs, examples)
    
    logger.info("All done! 🎉")


if __name__ == "__main__":
    main()