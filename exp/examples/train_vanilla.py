#!/usr/bin/env python3
"""
Train Vanilla Transformer with Hook System

Usage:
    From repository root: python -m exp.examples.train_vanilla [args]
"""

import sys
import os
import warnings

# suppress accelerate kernel version warnings globally
warnings.filterwarnings("ignore", message=".*kernel version.*")
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

from src.utils.training_utils import (
    create_base_parser, setup_training_environment, create_config_from_args,
    setup_data_loaders, setup_trainer_with_hooks, test_generation, log_if_main
)
from src.config.config import print_config
from src.mytokenizers import create_tokenizer
from src.model import get_model
import torch

def parse_args():
    """Parse vanilla-specific arguments."""
    parser = create_base_parser("Train Vanilla Transformer with Hook System")
    parser.add_argument('--output_dir', type=str, default='./outputs/vanilla_clean')
    return parser.parse_args()

def main():
    """Main vanilla training function."""
    args = parse_args()
    
    # setup env
    logger, device = setup_training_environment(args.output_dir, "Vanilla Transformer", args.trainer_type)
    
    # create config
    config = create_config_from_args(args)
    
    # init tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type)
    config.update_from_tokenizer(tokenizer)
    
    print_config(config, dataset_name=args.dataset)
    
    # setup data
    train_dataloader, val_dataloader, tokenizer = setup_data_loaders(args, config, tokenizer, logger, args.trainer_type)
    
    # create model
    log_if_main(logger, "Creating Vanilla Transformer...", args.trainer_type)
    model = get_model("vanilla", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_if_main(logger, f"Model: {num_params/1e6:.2f}M parameters", args.trainer_type)
    
    # setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    
    # create trainer with hooks
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "Vanilla"
    )
    
    # train
    log_if_main(logger, "Starting vanilla transformer training...", args.trainer_type)
    training_result = trainer.train()
    
    # test generation
    test_generation(model, tokenizer, device, args, logger, "vanilla", args.trainer_type)
    
    log_if_main(logger, "Vanilla transformer training completed!", args.trainer_type)

if __name__ == "__main__":
    main()