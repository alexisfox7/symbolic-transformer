import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.training_utils import (
    create_base_parser, setup_training_environment, create_config_from_args,
    setup_data_loaders, setup_trainer_with_hooks, test_generation, save_model_checkpoint
)
from config.config import print_config
from mytokenizers import create_tokenizer
from model import get_model
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
    logger, device = setup_training_environment(args.output_dir, "Vanilla Transformer")
    
    # create config
    config = create_config_from_args(args)
    
    # init tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type)
    config.update_from_tokenizer(tokenizer)
    
    print_config(config, dataset_name=args.dataset)
    
    # setup data
    train_dataloader, val_dataloader, tokenizer = setup_data_loaders(args, config, tokenizer, logger)
    
    # create model
    logger.info("Creating Vanilla Transformer...")
    model = get_model("vanilla", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params/1e6:.2f}M parameters")
    
    # setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    
    # create trainer with hooks
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "Vanilla"
    )
    
    # train
    logger.info("Starting vanilla transformer training...")
    training_result = trainer.train()
    
    # save model
    save_model_checkpoint(
        model, config, training_result, args.output_dir, "vanilla_model.pt", 
        logger=logger
    )
    
    # test generation
    test_generation(model, tokenizer, device, args, logger, "vanilla")
    
    logger.info("Vanilla transformer training completed!")

if __name__ == "__main__":
    main()