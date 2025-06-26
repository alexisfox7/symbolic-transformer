#!/usr/bin/env python3
"""
Train Token-Factored Transformer (TFT) with Hook System

TFT implements explicit stream separation (X = Xt + Xe) without vocabulary constraints,
providing mechanistic interpretability through attention-FFN coordination analysis.
"""

import sys
import os
import warnings

# Suppress accelerate kernel version warnings globally
warnings.filterwarnings("ignore", message=".*kernel version.*")
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from utils.training_utils import (
    create_base_parser, add_symbolic_args, setup_training_environment, 
    create_config_from_args, setup_data_loaders, setup_trainer_with_hooks, 
    test_generation
)
from config.config import print_config
from mytokenizers import create_tokenizer
from model import get_model
from model.architectures.tft import TFTTransformer
import torch

def parse_args():
    """Parse TFT-specific arguments."""
    parser = create_base_parser("Train Token-Factored Transformer (TFT) with Hook System")
    parser = add_symbolic_args(parser)  # TFT uses same symbolic flags (use_v, use_proj)
    parser.add_argument('--output_dir', type=str, default='./outputs/tft_clean')
    return parser.parse_args()

def main():
    """Main TFT training function."""
    args = parse_args()
    
    # Setup environment
    logger, device = setup_training_environment(args.output_dir, "Token-Factored Transformer", args.trainer_type)
    logger.info(f"TFT features: use_v={args.use_v}, use_proj={args.use_proj}")
    logger.info("Training TFT: stream separation without vocabulary constraints")
    
    # Create config with TFT features
    tft_features = {
        'use_v': args.use_v,
        'use_proj': args.use_proj
    }
    config = create_config_from_args(args, tft_features)
    
    # Initialize tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type)
    config.update_from_tokenizer(tokenizer)
    
    print_config(config, dataset_name=args.dataset)
    
    # Setup data
    train_dataloader, val_dataloader, tokenizer = setup_data_loaders(args, config, tokenizer, logger, args.trainer_type)
    
    # Create TFT model
    logger.info("Creating Token-Factored Transformer (TFT)...")
    model = get_model("tft", config=config).to(device)  # Use "tft" model type
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TFT Model: {num_params/1e6:.2f}M parameters")
    
    # Report TFT architectural features
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        first_block = model.transformer.h[0]
        if hasattr(first_block, 'attn'):
            logger.info(f"TFT Attention features: use_v={getattr(first_block.attn, 'use_v', False)}, "
                       f"use_proj={getattr(first_block.attn, 'use_proj', False)}")
        logger.info("TFT Architecture: Explicit stream separation (Xt + Xe) without vocabulary constraints")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    
    # Create trainer with hooks
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "TFT"
    )
    
    # Add TFT-specific state information
    trainer.trainer_state['tft_features'] = tft_features
    trainer.trainer_state['architecture_type'] = 'token_factored'
    
    # Train
    logger.info("Starting Token-Factored Transformer training...")
    logger.info("Architecture: X = Xt (symbolic stream) + Xe (contextual stream)")
    training_result = trainer.train()

    # Test generation
    test_generation(model, tokenizer, device, args, logger, "TFT", args.trainer_type)
    
    logger.info("Token-Factored Transformer training completed!")
    logger.info(f"TFT features used: use_v={args.use_v}, use_proj={args.use_proj}")
    logger.info("Stream separation achieved without vocabulary constraints")

if __name__ == "__main__":
    main()