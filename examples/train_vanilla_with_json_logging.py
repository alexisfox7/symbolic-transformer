#!/usr/bin/env python
# ./examples/train_vanilla_with_json_logging.py
"""
Simplified training script for Vanilla Transformer with JSON logging and validation.
Baseline comparison to symbolic transformer - no gradient accumulation complexity.

Usage:
    python examples/train_vanilla_with_json_logging.py --preset small --json_log_steps 50 --val_ratio 0.1
    python examples/train_vanilla_with_json_logging.py --preset medium --disable_json_logging --validate_every 2
    python examples/train_vanilla_with_json_logging.py --batch_size 32 --no_validation
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

# JSON logging imports
from utils.json_logger import create_json_logger_for_training
from trainers.json_trainer import create_accelerate_trainer_with_json_logging

# Validation imports
from torch.utils.data import DataLoader, random_split

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')

def parse_args():
    """Parse command line arguments with JSON logging and validation support."""
    parser = argparse.ArgumentParser(description='Train Vanilla Transformer with JSON Logging and Validation (Baseline)')
    
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
    
    # Trainer selection
    parser.add_argument("--trainer_type", type=str, default="accelerate",
                       choices=["simple", "accelerate"])
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # Tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'])
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./outputs/vanilla_json')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument("--save_model_filename", type=str, default="vanilla_model.pt")
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'])
    
    # Generation testing
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--generation_max_len", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=20)
    
    # JSON LOGGING ARGUMENTS
    parser.add_argument("--json_log_steps", type=int, default=100,
                       help="Log training metrics every N batches to JSON (default: 100)")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    parser.add_argument("--experiment_name", type=str, default="vanilla_transformer",
                       help="Experiment name for JSON logs")
    
    # VALIDATION ARGUMENTS
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1 = 10%)")
    parser.add_argument("--validate_every", type=int, default=1,
                       help="Validate every N epochs (default: 1)")
    parser.add_argument("--no_validation", action="store_true",
                       help="Disable validation (use full dataset for training)")
    
    return parser.parse_args()


def create_train_val_split(dataset, val_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Full dataset to split
        val_ratio: Fraction for validation (default: 0.1 = 10%)
        seed: Random seed for reproducible splits
        
    Returns:
        (train_dataset, val_dataset)
    """
    generator = torch.Generator().manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    logging.getLogger(__name__).info(f"Dataset split: {train_size} train, {val_size} validation")
    return train_dataset, val_dataset


def run_validation(model: torch.nn.Module, 
                  val_dataloader: DataLoader, 
                  device: torch.device):
    """
    Run validation and return metrics.
    
    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        device: Device to run on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.get('loss')
            
            if loss is not None and not torch.isnan(loss):
                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
    model.train()  # Return to training mode
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if not torch.isnan(torch.tensor(avg_loss)) else float('nan')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'samples': total_samples
    }


def load_and_prepare_data_with_validation(dataset_name, dataset_config, tokenizer, max_samples, 
                                        max_seq_length, batch_size, val_ratio=0.1, 
                                        mlm=False, split='train', shuffle=True):
    """
    Load data and create train/validation split.
    
    Args:
        val_ratio: Fraction for validation (default: 0.1)
        ... (other args same as original)
        
    Returns:
        (train_dataloader, val_dataloader, tokenizer)
    """
    logger = logging.getLogger(__name__)
    
    # Load full dataset
    full_dataloader, tokenizer = load_and_prepare_data(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        mlm=mlm,
        split=split,
        shuffle=False  # Don't shuffle yet, we'll split first
    )
    
    # Get the dataset from the dataloader
    full_dataset = full_dataloader.dataset
    
    # Create train/val split
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio)
    
    # Create separate dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=full_dataloader.collate_fn,
        drop_last=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        collate_fn=full_dataloader.collate_fn,
        drop_last=False,
        num_workers=0
    )
    
    logger.info(f"Created train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Created val dataloader: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader, tokenizer


class ValidationTrainerWrapper:
    """
    Wrapper to add validation to existing trainers.
    """
    def __init__(self, trainer, val_dataloader=None, validate_every=1, json_logger=None):
        self.trainer = trainer
        self.val_dataloader = val_dataloader
        self.validate_every = validate_every
        self.json_logger = json_logger
        
    def train(self):
        """Enhanced training with validation."""
        logger = logging.getLogger(__name__)
        
        # Store original methods
        original_log_epoch = self.trainer.log_epoch
        original_trigger_callbacks = self.trainer._trigger_callbacks
        
        validation_metrics = {
            'val_losses': [],
            'val_perplexities': []
        }
        
        def enhanced_log_epoch(epoch: int, avg_loss: float, metrics=None):
            """Enhanced epoch logging with validation."""
            # Call original logging
            original_log_epoch(epoch, avg_loss, metrics)
            
            # Run validation if needed
            if (self.val_dataloader and 
                epoch % self.validate_every == 0 and 
                hasattr(self.trainer, 'model') and 
                hasattr(self.trainer, 'device')):
                
                logger.info(f"Running validation for epoch {epoch}...")
                val_metrics = run_validation(self.trainer.model, self.val_dataloader, self.trainer.device)
                
                validation_metrics['val_losses'].append(val_metrics['loss'])
                validation_metrics['val_perplexities'].append(val_metrics['perplexity'])
                
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Log to JSON if available
                if self.json_logger:
                    # Check if we're in distributed training
                    is_main_process = True
                    if hasattr(self.trainer, 'accelerator'):
                        is_main_process = self.trainer.accelerator.is_main_process
                    
                    if is_main_process:
                        self.json_logger.log_validation(epoch, val_metrics)
                
                # Update metrics for callbacks
                if metrics is None:
                    metrics = {}
                metrics.update({
                    'val_loss': val_metrics['loss'],
                    'val_perplexity': val_metrics['perplexity']
                })
        
        def enhanced_trigger_callbacks(event_name: str, *args, **kwargs):
            """Enhanced callbacks with validation metrics."""
            # For epoch_end events, add validation metrics to logs
            if event_name == 'on_epoch_end' and len(args) >= 2:
                epoch = args[0]
                logs = args[1] if len(args) > 1 else kwargs.get('logs', {})
                
                # Add latest validation metrics if available
                if (validation_metrics['val_losses'] and 
                    epoch % self.validate_every == 0):
                    val_idx = (epoch // self.validate_every) - 1
                    if 0 <= val_idx < len(validation_metrics['val_losses']):
                        if isinstance(logs, dict):
                            logs.update({
                                'val_loss': validation_metrics['val_losses'][val_idx],
                                'val_perplexity': validation_metrics['val_perplexities'][val_idx]
                            })
            
            # Call original callbacks
            original_trigger_callbacks(event_name, *args, **kwargs)
        
        # Replace methods
        self.trainer.log_epoch = enhanced_log_epoch
        self.trainer._trigger_callbacks = enhanced_trigger_callbacks
        
        try:
            # Run training
            result = self.trainer.train()
            
            # Add validation metrics to result
            if isinstance(result, dict):
                result.update(validation_metrics)
                if validation_metrics['val_losses']:
                    result['final_val_loss'] = validation_metrics['val_losses'][-1]
                    result['final_val_perplexity'] = validation_metrics['val_perplexities'][-1]
            
            return result
            
        finally:
            # Restore original methods
            self.trainer.log_epoch = original_log_epoch
            self.trainer._trigger_callbacks = original_trigger_callbacks
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_vanilla_config(args):
    """Create configuration for the vanilla transformer."""
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


def load_checkpoint_for_resumption(checkpoint_path, model, optimizer, device, logger):
    """Load checkpoint for training resumption."""
    start_epoch = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state loaded successfully")
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded successfully")
            
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
            logger.warning(f"Checkpoint file not found. Starting from scratch.")
    
    return start_epoch


def main():
    """Main training function with JSON logging and validation."""
    args = parse_args()

    # Setup
    logger = setup_logging_and_output(args.output_dir)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*60)
    logger.info("VANILLA TRANSFORMER TRAINING WITH JSON LOGGING AND VALIDATION (BASELINE)")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Trainer: {args.trainer_type}")
    logger.info(f"JSON logging: {'Enabled' if not args.disable_json_logging else 'Disabled'}")
    if not args.disable_json_logging:
        logger.info(f"JSON log interval: every {args.json_log_steps} batches")
    
    # Validation info
    if not args.no_validation:
        logger.info(f"Validation: Enabled ({args.val_ratio:.1%} split, validate every {args.validate_every} epochs)")
    else:
        logger.info("Validation: Disabled")
    
    # Create configuration
    config = create_vanilla_config(args)
    
    # Initialize tokenizer
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
    
    # Setup JSON logging
    json_logger = None
    if not args.disable_json_logging:
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
                'model_type': 'vanilla_transformer',
            },
            'training_config': {
                'dataset': args.dataset,
                'max_samples': args.max_samples,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'trainer_type': args.trainer_type,
                'val_ratio': args.val_ratio if not args.no_validation else 0.0,
                'validate_every': args.validate_every,
            },
            'system_config': {
                'device': str(device),
                'tokenizer_type': args.tokenizer_type,
            }
        })
        logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
    # Load and prepare data WITH validation
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
        # New behavior - with validation split
        dataloader, val_dataloader, tokenizer = load_and_prepare_data_with_validation(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            max_seq_length=config.block_size,
            batch_size=config.batch_size,
            val_ratio=args.val_ratio,
            mlm=False,
            split='train',
            shuffle=True
        )
        logger.info(f"Data loaded. Train: {len(dataloader)} batches, Validation: {len(val_dataloader)} batches.")
    
    # Initialize model
    logger.info("Initializing Vanilla Transformer...")
    model = get_model("Vanilla", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {num_params/1e6:.2f}M parameters")
    
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
    
    # Create trainer with JSON logging
    logger.info(f"Setting up {args.trainer_type} trainer...")
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
            log_interval=args.log_interval
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
            log_interval=args.log_interval
        )
    
    # Wrap trainer with validation support
    if val_dataloader is not None:
        trainer = ValidationTrainerWrapper(
            trainer=trainer,
            val_dataloader=val_dataloader,
            validate_every=args.validate_every,
            json_logger=json_logger
        )
        logger.info("Trainer wrapped with validation support")
    
    # Adjust for resumption if needed
    if start_epoch > 0:
        remaining_epochs = config.num_epochs - start_epoch
        if remaining_epochs <= 0:
            logger.warning(f"No epochs remaining. Already completed {start_epoch} epochs.")
            return
        trainer.num_epochs = remaining_epochs
        logger.info(f"Adjusted training to {remaining_epochs} remaining epochs")
    
    # Train the model
    logger.info("="*60)
    logger.info(f"STARTING VANILLA TRANSFORMER TRAINING from epoch {start_epoch}")
    logger.info("="*60)
    
    training_result = trainer.train()
    
    logger.info("="*60)
    logger.info("VANILLA TRANSFORMER TRAINING COMPLETED")
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
        logger.info("TESTING VANILLA TRANSFORMER GENERATION")
        logger.info("="*60)
        
        test_prompts = [
            "The brave knight",
            "Once upon a time",
            "Spotty the dog loved",
            "The door was locked. Tim had a key.",
        ]
        
        model.eval()
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nTest {i+1}: '{prompt}'")
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
                logger.info(f"Generated: {generated_text}")
                
                # Log generation to JSON
                if json_logger:
                    # Check if we're using accelerate trainer
                    is_main_process = True
                    if hasattr(trainer, 'accelerator'):
                        is_main_process = trainer.accelerator.is_main_process
                    elif hasattr(trainer, 'trainer') and hasattr(trainer.trainer, 'accelerator'):
                        is_main_process = trainer.trainer.accelerator.is_main_process
                    
                    if is_main_process:
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
                logger.error(f"Error generating for '{prompt}': {e}")
    
    logger.info("\n" + "="*60)
    logger.info("VANILLA TRANSFORMER TRAINING COMPLETED!")
    logger.info("="*60)
    logger.info(f"Model: {num_params/1e6:.2f}M parameters")
    logger.info(f"Final training loss: {training_result.get('final_loss', 'N/A')}")
    if 'final_val_loss' in training_result:
        logger.info(f"Final validation loss: {training_result['final_val_loss']:.4f}")
        logger.info(f"Final validation perplexity: {training_result.get('final_val_perplexity', 'N/A')}")
    logger.info(f"Training time: {training_result.get('training_time', 'N/A')}")
    if json_logger:
        logger.info(f"JSON logs: {json_logger.log_file}")
    logger.info("BASELINE COMPLETE - Ready for comparison with symbolic transformer!")
    logger.info("="*60)


if __name__ == "__main__":
    main()