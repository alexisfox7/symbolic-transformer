#!/usr/bin/env python
# ./examples/train_vanilla.py
"""
Simple script to train a Vanilla Transformer with the same parameters as the Symbolic Transformer.
Uses the same argument structure as train_symbolic_example.py for easy comparison.
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset


def create_vanilla_config(args):
    """Create configuration for vanilla transformer using same structure as symbolic."""
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
    
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    config.generation_max_len = args.generation_max_len
    config.temperature = args.temperature
    config.top_k = args.top_k
    
    config.__post_init__()
    return config


def parse_args():
    """Parse command line arguments - same structure as symbolic script."""
    parser = argparse.ArgumentParser(description='Train Vanilla Transformer (Baseline)')
    
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    
    # Gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--effective_batch_size", type=int, default=None)
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'])
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/vanilla_baseline')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'])
    
    # Generation testing
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--generation_max_len", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    
    return parser.parse_args()


def load_checkpoint_for_resumption(checkpoint_path, model, optimizer, device, logger):
    """Load checkpoint for training resumption."""
    start_epoch = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state loaded successfully")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully")
            
            # Get starting epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resuming from epoch {start_epoch}")
            
            if 'loss' in checkpoint:
                logger.info(f"Checkpoint loss: {checkpoint['loss']:.6f}")
                
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning("Starting training from scratch")
            start_epoch = 0
    else:
        if checkpoint_path:
            logger.warning(f"Checkpoint file not found at '{checkpoint_path}'. Starting training from scratch.")
    
    return start_epoch


class ResumeTrainer:
    """Wrapper to handle resumption with the existing trainer system."""
    def __init__(self, trainer, start_epoch=0):
        self.trainer = trainer
        self.current_epoch = start_epoch
        
    def train(self):
        """Modified training loop that supports resumption."""
        original_num_epochs = self.trainer.num_epochs
        remaining_epochs = original_num_epochs - self.current_epoch
        
        if remaining_epochs <= 0:
            logger = logging.getLogger(__name__)
            logger.warning(f"No epochs remaining to train. Already completed {self.current_epoch} epochs.")
            return {'final_loss': 0.0, 'training_time': 0.0}
        
        self.trainer.num_epochs = remaining_epochs
        
        # Modify save_checkpoint to use correct epoch numbers
        original_save_checkpoint = self.trainer.save_checkpoint
        
        def adjusted_save_checkpoint(path, epoch=None, **kwargs):
            adjusted_epoch = epoch + self.current_epoch if epoch is not None else None
            if epoch is not None and self.trainer.output_dir:
                filename = f"checkpoint_epoch_{adjusted_epoch}.pt"
                path = os.path.join(self.trainer.output_dir, filename)
            return original_save_checkpoint(path, adjusted_epoch, **kwargs)
        
        self.trainer.save_checkpoint = adjusted_save_checkpoint
        
        # Run training
        result = self.trainer.train()
        
        # Restore original values
        self.trainer.num_epochs = original_num_epochs
        self.trainer.save_checkpoint = original_save_checkpoint
        
        return result


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("VANILLA TRANSFORMER TRAINING (BASELINE)")
    print("="*60)
    
    if args.resume_from_checkpoint:
        logger.info(f"Checkpoint resumption requested: {args.resume_from_checkpoint}")
    
    # Create config
    config = create_vanilla_config(args)
    
    # Initialize tokenizer (same logic as symbolic script)
    if args.tokenizer_type == "character":
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
    
    config.update_from_tokenizer(tokenizer)
    
    # Print config
    print_config(config, dataset_name=args.dataset, model=model)
    
    # Load data
    logger.info("Loading data...")
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
    
    # Create model
    logger.info("Creating Vanilla Transformer...")
    model = get_model("Vanilla", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params/1e6:.2f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Load checkpoint if resuming
    start_epoch = load_checkpoint_for_resumption(
        args.resume_from_checkpoint, model, optimizer, device, logger
    )
    
    # Setup trainer
    trainer = get_trainer(
        trainer_type="simple",
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        output_dir=args.output_dir,
        clip_grad_norm=args.clip_grad_norm,
        log_interval=256,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.effective_batch_size
    )
    
    # Wrap trainer for resumption support
    resume_trainer = ResumeTrainer(trainer, start_epoch)
    
    # Train
    logger.info(f"Starting training from epoch {start_epoch}...")
    training_result = resume_trainer.train()
    
    # Save model
    model_path = os.path.join(args.output_dir, "vanilla_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.num_epochs,
        'config': config,
        'tokenizer': tokenizer,
        'training_result': training_result,
        'training_args': vars(args),
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test generation
    if not args.skip_generation:
        test_prompts = [
            "The brave knight",
            "Once upon a time", 
            "Spotty the dog",
            "The little girl"
        ]
        
        print("\nGeneration test:")
        model.eval()
        for prompt in test_prompts:
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
                print(f"'{prompt}' → '{generated_text}'")
            except Exception as e:
                print(f"'{prompt}' → Error: {e}")
    
    print("\nVanilla training complete!")


if __name__ == "__main__":
    main()