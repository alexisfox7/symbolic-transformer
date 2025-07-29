#!/usr/bin/env python3
"""
Test generation from a trained checkpoint.
Based on checkpoint parsing from run_inference_with_hooks.

Usage:
    python test_generation.py path/to/checkpoint.pt --prompt "Once upon a time"
"""

import torch
import argparse
import os
import sys
from accelerate import PartialState

from mytokenizers.factory import add_reasoning_tokens
from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.mytokenizers import create_tokenizer, from_pretrained
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device, arg_model_type):
    """Load model from checkpoint (adapted from run_inference_with_hooks)."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        # handle both dict and TransformerConfig object
        if hasattr(config_data, '__dict__'):
            config = config_data 
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    model_type = arg_model_type 
    logger.info(f"Model type: {model_type}")
    model = get_model(model_type, config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # try loading directly if it's just the state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Test generation from trained checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='vanilla', 
                       choices=['vanilla', 'tft', 'symbolic'],
                       help='Model architecture type')
    parser.add_argument('--prompt', type=str, default="The girl said",
                       help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer type or path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Initialize accelerate state for logging
    PartialState()
    
    # load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    
    # create tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = from_pretrained(args.tokenizer)
    else:
        tokenizer = create_tokenizer(args.tokenizer)
        tokenizer = add_reasoning_tokens(tokenizer)
    
    logger.info(f"Running generation test with {args.num_samples} samples")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Parameters: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}")
    logger.info(f"{'='*60}")
    
    # generate multiple samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            logger.info(f"\nSample {i+1}/{args.num_samples}:")
            logger.info("-" * 40)
        
        try:
            ids, generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=args.prompt,
                device=device,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                hooks=[]  # no hooks for simple generation test
            )
            
            print(generated_text)
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.info("Generation test completed!")


if __name__ == "__main__":
    main()