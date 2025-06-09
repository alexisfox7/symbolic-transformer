#!/usr/bin/env python
# ./examples/train_symbolic_with_json_logging.py
"""
Training script for Symbolic Transformer with built-in JSON logging support.
FIXED: Only prints from main process in distributed training.
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
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset

# JSON logging imports
from utils.json_logger import create_json_logger_for_training
from trainers.json_trainer import create_accelerate_trainer_with_json_logging


class DistributedLogger:
    """Logger that only prints from main process in distributed training."""
    
    def __init__(self, accelerator=None):
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process if accelerator else True
        
    def info(self, message):
        if self.is_main_process:
            print(message)
            
    def warning(self, message):
        if self.is_main_process:
            print(f"WARNING: {message}")
            
    def error(self, message):
        if self.is_main_process:
            print(f"ERROR: {message}")


def parse_args():
    """Parse command line arguments with JSON logging support."""
    parser = argparse.ArgumentParser(description='Train Symbolic Transformer with JSON Logging')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use")
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size preset')
    parser.add_argument('--block_size', type=int, default=None,
                       help='Training sequence length')
    parser.add_argument("--n_layer", type=int, default=None, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=None, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability")
    parser.add_argument("--bias", action='store_true', help="Use bias in linear layers")
    
    # Symbolic-specific parameters
    parser.add_argument("--use_symbolic_ffn", action='store_true', default=True,
                       help="Use vocabulary-constrained FFN")
    parser.add_argument("--no_symbolic_ffn", action='store_false', dest='use_symbolic_ffn',
                       help="Disable symbolic FFN")
    parser.add_argument("--use_vocab_refinement", action='store_true', default=False,
                       help="Use vocabulary refinement in projections")
    parser.add_argument("--use_v", action='store_true', default=False,
                       help="Use value projection in attention")
    parser.add_argument("--use_proj", action='store_true', default=False,
                       help="Use output projection in attention")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Mini-batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, 
                       help="Max norm for gradient clipping")
    
    # Trainer selection
    parser.add_argument("--trainer_type", type=str, default="accelerate",
                       choices=["simple", "accelerate"],
                       help="Type of trainer to use")
    
    # Gradient accumulation parameters
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of mini-batches to accumulate")
    parser.add_argument("--effective_batch_size", type=int, default=None,
                       help="Target effective batch size")
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'],
                       help='Type of tokenizer to use')
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./outputs/symbolic_json',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=256,
                       help='Logging interval')
    parser.add_argument("--save_model_filename", type=str, default="symbolic_model.pt",
                       help="Filename for the saved model")
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    # Generation testing
    parser.add_argument("--skip_generation", action="store_true", 
                       help="Skip sample text generation after training")
    parser.add_argument("--generation_max_len", type=int, default=30, 
                       help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.5, 
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, 
                       help="Top-k sampling parameter")
    
    # JSON LOGGING ARGUMENTS
    parser.add_argument("--json_log_steps", type=int, default=256,
                       help="Log training metrics every N steps to JSON (default: 256)")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    parser.add_argument("--experiment_name", type=str, default="symbolic_transformer",
                       help="Experiment name for JSON logs")
    
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


def setup_logging_and_output(output_dir, is_main_process):
    """Setup logging and output directory - only on main process."""
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup basic logging only on main process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    return logging.getLogger(__name__)


def load_checkpoint_for_resumption(checkpoint_path, model, optimizer, device, logger, is_main_process):
    """Load checkpoint for training resumption."""
    start_epoch = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        if is_main_process:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if is_main_process:
                    logger.info("Model state loaded successfully")
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if is_main_process:
                    logger.info("Optimizer state loaded successfully")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if is_main_process:
                    logger.info(f"Resuming from epoch {start_epoch}")
                
            if 'loss' in checkpoint and is_main_process:
                logger.info(f"Checkpoint loss: {checkpoint['loss']:.6f}")
                
        except Exception as e:
            if is_main_process:
                logger.error(f"Error loading checkpoint: {e}")
                logger.warning("Starting training from scratch")
            start_epoch = 0
    else:
        if checkpoint_path and is_main_process:
            logger.warning(f"Checkpoint file not found. Starting from scratch.")
    
    return start_epoch


def main():
    """Main training function with JSON logging - FIXED for distributed training."""
    args = parse_args()
    
    # Create a temporary trainer to get accelerator for distributed setup
    temp_trainer = None
    accelerator = None
    is_main_process = True
    
    if args.trainer_type == "accelerate":
        try:
            from trainers.accelerate_trainer import AccelerateTrainer
            temp_trainer = AccelerateTrainer(
                model=torch.nn.Linear(1, 1),  # Dummy model
                dataloader=None,
                optimizer=None,
                device=torch.device('cpu'),
                num_epochs=1
            )
            accelerator = temp_trainer.accelerator
            is_main_process = accelerator.is_main_process
        except:
            pass  # Fall back to single process
    
    # Create distributed logger
    dist_logger = DistributedLogger(accelerator)
    
    # Setup only on main process
    logger = setup_logging_and_output(args.output_dir, is_main_process)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Only print from main process
    if is_main_process:
        dist_logger.info("="*60)
        dist_logger.info("SYMBOLIC TRANSFORMER TRAINING WITH JSON LOGGING")
        dist_logger.info("="*60)
        dist_logger.info(f"Device: {device}")
        dist_logger.info(f"Trainer: {args.trainer_type}")
        if accelerator:
            dist_logger.info(f"Distributed training: {accelerator.num_processes} processes")
            dist_logger.info(f"Process rank: {accelerator.process_index}")
        dist_logger.info(f"JSON logging: {'Enabled' if not args.disable_json_logging else 'Disabled'}")
        if not args.disable_json_logging:
            dist_logger.info(f"JSON log interval: every {args.json_log_steps} steps")
    
    # Create configuration
    config = create_symbolic_config(args)
    
    # Initialize tokenizer
    if is_main_process:
        dist_logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
        
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
                if is_main_process:
                    dist_logger.error(f"Could not find text column. Available: {temp_dataset.column_names}")
                sys.exit(1)
            text_samples = temp_dataset[text_field]
        
        tokenizer = create_tokenizer(args.tokenizer_type)
        tokenizer.build_vocab_from_texts([str(item) for item in text_samples])
    else:
        tokenizer = create_tokenizer(args.tokenizer_type)
    
    # Update config with tokenizer info
    config.update_from_tokenizer(tokenizer)
    
    # Print configuration only on main process
    if is_main_process:
        print_config(config, dataset_name=args.dataset)
    
    # Setup JSON logging (only on main process)
    json_logger = None
    if not args.disable_json_logging and is_main_process:
        json_logger = create_json_logger_for_training(
            args.output_dir, 
            args.experiment_name, 
            args.json_log_steps
        )
        
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
                'trainer_type': args.trainer_type,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'effective_batch_size': args.effective_batch_size,
            },
            'system_config': {
                'device': str(device),
                'tokenizer_type': args.tokenizer_type,
                'num_processes': accelerator.num_processes if accelerator else 1,
                'process_index': accelerator.process_index if accelerator else 0,
            }
        })
        dist_logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    # Load and prepare data
    if is_main_process:
        dist_logger.info("Loading and preparing data...")
        
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
    
    if is_main_process:
        dist_logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")
    
    # Initialize model
    if is_main_process:
        dist_logger.info("Initializing Symbolic Transformer...")
        
    model = get_model("Symbolic", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if is_main_process:
        dist_logger.info(f"Model initialized with {num_params/1e6:.2f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint_for_resumption(
        args.resume_from_checkpoint, model, optimizer, device, logger, is_main_process
    )
    
    # Create trainer with JSON logging
    if is_main_process:
        dist_logger.info(f"Setting up {args.trainer_type} trainer...")
        
    if args.trainer_type == "accelerate":
        trainer = create_accelerate_trainer_with_json_logging(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            json_logger=json_logger,
            num_epochs=config.num_epochs,
            output_dir=args.output_dir,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
    else:
        # Simple trainer fallback
        trainer = get_trainer(
            trainer_type=args.trainer_type,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=config.num_epochs,
            output_dir=args.output_dir,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            effective_batch_size=args.effective_batch_size
        )
    
    # Adjust for resumption if needed
    if start_epoch > 0:
        remaining_epochs = config.num_epochs - start_epoch
        if remaining_epochs <= 0:
            if is_main_process:
                dist_logger.warning(f"No epochs remaining. Already completed {start_epoch} epochs.")
            return
        trainer.num_epochs = remaining_epochs
        if is_main_process:
            dist_logger.info(f"Adjusted training to {remaining_epochs} remaining epochs")
    
    # Train the model
    if is_main_process:
        dist_logger.info("="*60)
        dist_logger.info(f"STARTING TRAINING from epoch {start_epoch}")
        dist_logger.info("="*60)
    
    training_result = trainer.train()
    
    if is_main_process:
        dist_logger.info("="*60)
        dist_logger.info("TRAINING COMPLETED")
        dist_logger.info("="*60)
    
    # Save final model (only main process)
    if is_main_process:
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
        dist_logger.info(f"Model saved to {model_path}")
    
    # Test generation (only main process)
    if not args.skip_generation and is_main_process:
        dist_logger.info("="*60)
        dist_logger.info("TESTING SYMBOLIC GENERATION")
        dist_logger.info("="*60)
        
        test_prompts = [
            "The brave knight",
            "Once upon a time",
            "Spotty the dog loved",
            "The door was locked. Tim had a key.",
        ]
        
        model.eval()
        for i, prompt in enumerate(test_prompts):
            dist_logger.info(f"\nTest {i+1}: '{prompt}'")
            try:
                _, generated_text = run_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt,
                    device=device,
                    max_new_tokens=args.generation_max_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    show_progress=False
                )
                dist_logger.info(f"Generated: {generated_text}")
                
                # Log generation to JSON
                if json_logger:
                    json_logger.log_generation(
                        epoch=config.num_epochs,
                        prompt=prompt,
                        generated=generated_text,
                        generation_params={
                            'max_new_tokens': args.generation_max_len,
                            'temperature': args.temperature,
                            'top_k': args.top_k
                        }
                    )
                        
            except Exception as e:
                dist_logger.error(f"Error generating for '{prompt}': {e}")
    
    # Final summary (only main process)
    if is_main_process:
        dist_logger.info("\n" + "="*60)
        dist_logger.info("SYMBOLIC TRANSFORMER TRAINING COMPLETED!")
        dist_logger.info("="*60)
        dist_logger.info(f"Model: {num_params/1e6:.2f}M parameters")
        dist_logger.info(f"Final loss: {training_result.get('final_loss', 'N/A')}")
        dist_logger.info(f"Training time: {training_result.get('training_time', 'N/A')}")
        if json_logger:
            dist_logger.info(f"JSON logs: {json_logger.log_file}")
        if accelerator:
            dist_logger.info(f"Trained on {accelerator.num_processes} processes")
        dist_logger.info("="*60)


if __name__ == "__main__":
    main()