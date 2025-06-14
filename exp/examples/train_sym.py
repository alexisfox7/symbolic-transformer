#!/usr/bin/env python
# examples/train_symbolic_with_json_logging.py - FIXED VERSION
"""
Fixed version with simplified JSON logging.
No more dual loggers or complex method overriding.
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from modelold import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset

# SIMPLIFIED JSON logging - replace complex imports
from utils.json_logger3 import create_simple_json_logger, SimpleJSONTrainerWrapper

# Validation imports
from torch.utils.data import DataLoader, random_split

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Symbolic Transformer with Simple JSON Logging')
    
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
    parser.add_argument('--trainer_type', type=str, default='accelerate', 
                       choices=['simple', 'accelerate'])
    
    # Generation parameters
    parser.add_argument('--generation_max_len', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--skip_generation', action='store_true')
    
    # Tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='character',
                       choices=['character', 'word', 'bpe'])
    
    # Output and checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs/symbolic_training')
    parser.add_argument('--save_model_filename', type=str, default='final_model.pt')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    
    # SIMPLIFIED JSON logging arguments
    parser.add_argument("--json_log_steps", type=int, default=100,
                       help="Log to JSON every N steps")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging completely")
    parser.add_argument("--experiment_name", type=str, default="symbolic_experiment",
                       help="Name for this experiment")
    
    # Validation arguments
    parser.add_argument("--no_validation", action="store_true",
                       help="Disable validation")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--validate_every", type=int, default=1,
                       help="Validate every N epochs")
    
    # Additional training options
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    
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


def setup_logging_and_output(output_dir):
    """Setup logging and output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training function with simplified JSON logging."""
    args = parse_args()

    # Setup
    logger = setup_logging_and_output(args.output_dir)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*60)
    logger.info("SYMBOLIC TRANSFORMER TRAINING WITH SIMPLE JSON LOGGING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Trainer: {args.trainer_type}")
    logger.info(f"JSON logging: {'Enabled' if not args.disable_json_logging else 'Disabled'}")
    if not args.disable_json_logging:
        logger.info(f"JSON log interval: every {args.json_log_steps} steps")
    
    # Create configuration
    config = create_symbolic_config(args)
    
    # Initialize tokenizer (keeping original logic)
    logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
    if args.tokenizer_type == "character":
        # Build character vocab from dataset sample
        temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
        temp_dataset_args = [args.dataset]
        if args.dataset_config:
            temp_dataset_args.append(args.dataset_config)
        
        temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
        
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
    
    # SIMPLIFIED JSON logging setup
    json_logger = None
    if not args.disable_json_logging:
        json_logger = create_simple_json_logger(
            args.output_dir, 
            args.experiment_name, 
            args.json_log_steps
        )
        
        # Log configuration once
        if json_logger:
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
                    'trainer_type': args.trainer_type,
                },
                'system_config': {
                    'device': str(device),
                    'tokenizer_type': args.tokenizer_type,
                }
            })
            logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    # Load and prepare data (keeping original validation logic)
    logger.info("Loading and preparing data...")
    if args.no_validation:
        # Original behavior - no validation split
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
        val_dataloader = None
        logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches (no validation).")
    else:
        # Load data with validation split
        full_dataloader, tokenizer = load_and_prepare_data(
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
        
        # Split into train/validation
        dataset = full_dataloader.dataset
        val_size = int(len(dataset) * args.val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        logger.info(f"Data loaded. Train: {len(dataloader)} batches, Validation: {len(val_dataloader)} batches.")
    
    # Create model
    logger.info("Creating model...")
    model = get_model("symbolic", config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create trainer - MUST use original method for identical behavior
    logger.info(f"Creating {args.trainer_type} trainer...")
    if args.trainer_type == "accelerate" and json_logger:
        # Use ORIGINAL trainer creation for accelerate + JSON logging
        from trainers.json_trainer import create_accelerate_trainer_with_json_logging
        trainer = create_accelerate_trainer_with_json_logging(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            json_logger=json_logger,
            num_epochs=config.num_epochs,
            log_interval=args.log_interval,
            clip_grad_norm=args.clip_grad_norm
        )
    else:
        # Use standard trainer creation
        trainer = get_trainer(
            trainer_type=args.trainer_type,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=config.num_epochs,
            log_interval=args.log_interval,
            clip_grad_norm=args.clip_grad_norm
        )
        
        # Add simple JSON logging for non-accelerate trainers
        if json_logger:
            trainer = SimpleJSONTrainerWrapper(trainer, json_logger)
    
    # Train the model
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    training_result = trainer.train()
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    
    # Save final model (keeping original logic)
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
    
    # Test generation (keeping original logic)
    if not args.skip_generation:
        logger.info("="*60)
        logger.info("TESTING SYMBOLIC GENERATION")
        logger.info("="*60)
        
        test_prompts = [
            "The brave knight",
            "Once upon a time",
            "Spotty the dog loved",
            "The door was locked. Tim had a key."
        ]
        
        for prompt in test_prompts:
            try:
                result = run_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=config.generation_max_len,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    device=device
                )
                logger.info(f"Prompt: '{prompt}' -> Generated: '{result['generated_text']}'")
                
                # Log generation to JSON
                if json_logger:
                    json_logger._write_log({
                        "event": "generation",
                        "prompt": prompt,
                        "generated": result['generated_text']
                    })
                    
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")

    logger.info("="*60)
    logger.info("SYMBOLIC TRAINING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()