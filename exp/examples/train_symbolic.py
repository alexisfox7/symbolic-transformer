#!/usr/bin/env python3
"""
Train Symbolic Transformer with Hook System

Usage:
    From repository root: python -m exp.examples.train_symbolic [args]
"""

import sys
import os
import warnings

# suppress accelerate kernel version warnings globally
warnings.filterwarnings("ignore", message=".*kernel version.*")
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

from src.utils.training_utils import (
    create_base_parser, add_symbolic_args, setup_data_loaders_with_combined, setup_training_environment, 
    create_config_from_args, setup_data_loaders, setup_trainer_with_hooks, 
    test_generation, log_if_main
)
from src.config.config import print_config
from src.mytokenizers import create_tokenizer
from src.model import get_model
import torch

def parse_args():
    """Parse symbolic-specific arguments."""
    parser = create_base_parser("Train Symbolic Transformer with Hook System")
    parser = add_symbolic_args(parser) 
    parser.add_argument('--output_dir', type=str, default='./outputs/symbolic_clean')
    parser.add_argument('--vocab-ffn', action='store_true', default=True,
                        help='Use vocabulary-constrained FFN (SymbolicFFN). If False, uses VanillaFFN')
    return parser.parse_args()

def main():
    """Main symbolic training function."""
    args = parse_args()
    
    # setup environment
    logger, device = setup_training_environment(args.output_dir, "Symbolic Transformer", args.trainer_type)
    log_if_main(logger, f"Symbolic features: use_v={args.use_v}, use_proj={args.use_proj}, vocab_ffn={args.vocab_ffn}", args.trainer_type)
    
    # create config
    symbolic_features = {
        'use_v': args.use_v,
        'use_proj': args.use_proj,
        'vocab_ffn': args.vocab_ffn
    }
    config = create_config_from_args(args, symbolic_features)
    
    # init tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type)
    config.update_from_tokenizer(tokenizer)
    
    print_config(config, dataset_name=args.dataset)
    
    # setup data
    #train_dataloader, val_dataloader, tokenizer = setup_data_loaders(args, config, tokenizer, logger, args.trainer_type)
    train_dataloader, val_dataloader, tokenizer = setup_data_loaders_with_combined(args, config, tokenizer, logger, args.trainer_type)
    
    # create model
    log_if_main(logger, "Creating Symbolic Transformer...", args.trainer_type)
    model = get_model("symbolic", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_if_main(logger, f"Model: {num_params/1e6:.2f}M parameters", args.trainer_type)
    
    # report symbolic features in model
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        first_block = model.transformer.h[0]
        if hasattr(first_block, 'attn'):
            log_if_main(logger, f"Attention features: use_v={getattr(first_block.attn, 'use_v', False)}, "
                       f"use_proj={getattr(first_block.attn, 'use_proj', False)}", args.trainer_type)
    
    # setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    
    # create trainer with hooks
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "Symbolic"
    )
    
    # train
    log_if_main(logger, "Starting symbolic transformer training...", args.trainer_type)
    training_result = trainer.train()

    # test generation
    test_generation(model, tokenizer, device, args, logger, "symbolic", args.trainer_type)
    
    log_if_main(logger, "Symbolic transformer training completed!", args.trainer_type)
    log_if_main(logger, f"Symbolic features used: use_v={args.use_v}, use_proj={args.use_proj}, vocab_ffn={args.vocab_ffn}", args.trainer_type)

if __name__ == "__main__":
    main()