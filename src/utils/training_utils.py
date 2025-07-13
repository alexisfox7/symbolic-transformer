#examples/training_utils.py
"""
Shared utilities for training scripts.
Reduces duplication between vanilla and tft/symbolic training.
"""

import argparse
import os
import sys
import torch
import logging
from torch.utils.data import DataLoader, random_split
import warnings

# suppress accelerate kernel version warnings globally
warnings.filterwarnings("ignore", message=".*kernel version.*")
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

from src.config.config import get_preset_config
from src.utils.data_utils import load_and_prepare_data

def log_if_main(logger, message, trainer_type="simple"):
    """Log only from main process when using accelerate."""
    if trainer_type == "accelerate":
        try:
            import os
            # Check if we're in a distributed setting
            if os.environ.get('LOCAL_RANK', '0') == '0':
                logger.info(message)
        except:
            pass
    else:
        logger.info(message)

def create_base_parser(description="Train Transformer with Hook System"):
    """Create base argument parser with common arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    # dataset 
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    
    # model config 
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    
    # training 
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--trainer_type", type=str, default="simple",
                       choices=["simple", "accelerate"])
    
    # logging & validation 
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--json_log_steps", type=int, default=100)
    parser.add_argument("--disable_json_logging", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--validate_every", type=int, default=1)
    parser.add_argument("--no_validation", action="store_true")
    
    # output 
    parser.add_argument('--tokenizer_type', type=str, default='gpt2')
    
    # generation testing 
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--generation_max_len", type=int, default=30)
    
    # attention parameters
    parser.add_argument("--use_sparsemax", action="store_true", default=False,
                       help="Use sparsemax instead of softmax in attention")
    
    return parser

def add_symbolic_args(parser):
    """Add symbolic-specific arguments to parser."""
    parser.add_argument("--use_v", type=str, default="none",
                       choices=["none", "normal", "kronecker"],
                       help="V matrix parameterization type in attention")
    parser.add_argument("--use_proj", type=str, default="none",
                       choices=["none", "normal", "kronecker"],
                       help="Output projection parameterization type")
    return parser

def setup_training_environment(output_dir, model_type="Transformer", trainer_type="simple"):
    """Setup logging and output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Only configure logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_if_main(logger, "="*60, trainer_type)
    log_if_main(logger, f"{model_type.upper()} TRAINING WITH HOOK SYSTEM", trainer_type)
    log_if_main(logger, "="*60, trainer_type)
    
    return logger, device

def create_config_from_args(args, symbolic_features=None):
    """Create config from parsed arguments."""
    config = get_preset_config(args.preset)
    
    # override with command line arguments
    if args.n_embd: config.n_embd = args.n_embd
    if args.n_head: config.n_head = args.n_head
    if args.n_layer: config.n_layer = args.n_layer
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.use_sparsemax = args.use_sparsemax
    
    if symbolic_features:
        for feature, value in symbolic_features.items():
            setattr(config, feature, value)
    
    return config

def create_train_val_split(dataset, val_ratio=0.1, seed=42):
    """Create train/validation split."""
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def setup_data_loaders(args, config, tokenizer, logger, trainer_type="simple"):
    """Setup training and validation data loaders."""
    log_if_main(logger, "Loading data...", trainer_type)
    
    # load full dataset
    full_dataloader, tokenizer = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=config.block_size,
        batch_size=config.batch_size,
        mlm=False, split='train', shuffle=False
    )
    
    # create train/val split if enabled
    if not args.no_validation:
        train_dataset, val_dataset = create_train_val_split(
            full_dataloader.dataset, args.val_ratio
        )
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=full_dataloader.collate_fn, drop_last=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=full_dataloader.collate_fn, drop_last=False
        )
        log_if_main(logger, f"Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches", trainer_type)
    else:
        train_dataloader = full_dataloader
        val_dataloader = None
        log_if_main(logger, f"Training: {len(train_dataloader)} batches (no validation)", trainer_type)
    
    return train_dataloader, val_dataloader, tokenizer

def run_validation(model, val_dataloader, device):
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            batch = {k: v.to(device) for k, v in batch_data.items() 
                    if isinstance(v, torch.Tensor)}
            
            outputs = model(**batch)
            loss = outputs.get('loss')
            
            if loss is not None and not torch.isnan(loss):
                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
    model.train()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if not torch.isnan(torch.tensor(avg_loss)) else float('nan')
    
    return {'loss': avg_loss, 'perplexity': perplexity, 'samples': total_samples}

def setup_trainer_with_hooks(trainer_type, model, train_dataloader, optimizer, device, 
                           config, args, val_dataloader=None, model_type=""):
    """Setup trainer with standard hooks."""
    from trainers import get_trainer
    from trainers.hooks import ValidationHook
    
    trainer = get_trainer(
        trainer_type=trainer_type,
        model=model, dataloader=train_dataloader, optimizer=optimizer, device=device,
        num_epochs=config.num_epochs, output_dir=args.output_dir,
        clip_grad_norm=args.clip_grad_norm, log_interval=args.log_interval
    )
    
    trainer.trainer_state['config'] = config
    trainer.trainer_state['model_type'] = model_type
    
    # add validation hook FIRST so other hooks can use the metrics
    if val_dataloader:
        validation_hook = ValidationHook(val_dataloader, device, args.validate_every, model_type)
        trainer.add_hook(validation_hook)
    
    # add standard hooks
    trainer.add_console_logging(log_every_n_batches=args.log_interval)
    
    if not args.disable_json_logging:
        trainer.add_json_logging(log_every_n_batches=args.json_log_steps)
        
    trainer.add_checkpointing(save_every_n_epochs=1)
    
    return trainer

def test_generation(model, tokenizer, device, args, logger, model_type="", trainer_type="simple"):
    """Test model generation with sample prompts."""
    if args.skip_generation:
        return
        
    from inference.generation import run_generation
    
    log_if_main(logger, f"Testing {model_type.lower()} generation...", trainer_type)
    test_prompts = ["The brave knight", "Once upon a time", "The magical forest"]
    
    model.eval()
    for prompt in test_prompts:
        try:
            _, generated_text = run_generation(
                model=model, tokenizer=tokenizer, prompt_text=prompt,
                device=device, max_new_tokens=args.generation_max_len
            )
            log_if_main(logger, f"'{prompt}' → '{generated_text}'", trainer_type)
        except Exception as e:
            logger.error(f"Generation failed for '{prompt}': {e}")