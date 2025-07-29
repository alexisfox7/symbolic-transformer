#!/usr/bin/env python3
"""
Run logit lens analysis on a trained model.

Usage:
    python -m run_logit_lens --checkpoint path/to/checkpoint.pt --text "The cat sat on the"
"""

import argparse
import torch
import os
import sys
import logging

from src.model import get_model
from src.config import TransformerConfig
from src.mytokenizers import create_tokenizer, from_pretrained
from src.inference.logit_lens import run_logit_lens_analysis, plot_logit_lens, print_logit_lens_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_symbolic_with_config_detection(checkpoint_path, device='cpu'):
    """Load symbolic model with automatic config detection from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        if hasattr(config_data, '__dict__'):
            config = config_data
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    # Auto-detect symbolic features from state_dict
    state_dict_keys = set(checkpoint['model_state_dict'].keys())
    
    # Check for Kronecker features
    has_v_tmp = any('v_tmp' in key for key in state_dict_keys)
    has_proj_tmp = any('proj_tmp' in key for key in state_dict_keys)
    has_c_attn = any('c_attn' in key for key in state_dict_keys)
    
    print(f"Checkpoint features detected:")
    print(f"  - Kronecker V matrix (v_tmp): {has_v_tmp}")
    print(f"  - Kronecker projection (proj_tmp): {has_proj_tmp}")
    print(f"  - Standard attention (c_attn): {has_c_attn}")
    
    # Update config based on detected features
    if has_v_tmp:
        config.use_v = "kronecker"
        print("  → Setting use_v = 'kronecker'")
    elif has_c_attn:
        config.use_v = "normal"
        print("  → Setting use_v = 'normal'")
    else:
        config.use_v = "none"
        print("  → Setting use_v = 'none'")
    
    if has_proj_tmp:
        config.use_proj = "kronecker"
        print("  → Setting use_proj = 'kronecker'")
    else:
        config.use_proj = "none"
        print("  → Setting use_proj = 'none'")
    
    # Create model with corrected config
    print(f"\nCreating symbolic model with: use_v={config.use_v}, use_proj={config.use_proj}")
    model = get_model("symbolic", config)
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Perfect match - loaded successfully!")
    except RuntimeError as e:
        print("⚠️ Still some mismatch, loading with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ Loaded with partial match")
    
    model.to(device)
    model.eval()
    
    return model, config

def load_model_from_checkpoint(checkpoint_path, device, model_type):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        if hasattr(config_data, '__dict__'):
            config = config_data 
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    logger.info(f"Model type: {model_type}")
    model = get_model(model_type, config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Run logit lens analysis on a trained model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='vanilla', 
                       choices=['vanilla', 'symbolic', 'tft'],
                       help='Type of model architecture')
    parser.add_argument('--text', type=str, default="Once upon a time there was a girl. The young pretty girl was named",
                       help='Text to analyze')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--output-dir', type=str, default='outputs/logit_lens',
                       help='Directory to save results')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    #model, config = load_symbolic_with_config_detection(args.checkpoint, device)

    # Create tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = from_pretrained(args.tokenizer)
    else:
        tokenizer = create_tokenizer(args.tokenizer)
    
    # Run logit lens analysis
    logger.info(f"Running logit lens analysis on: '{args.text}'")
    predictions, final_prediction = run_logit_lens_analysis(model, tokenizer, args.text, device)
    
    if not predictions:
        logger.error("No predictions generated. Check if the model forward pass includes logit lens hooks.")
        return
    
    # Print results
    print_logit_lens_analysis(predictions, final_prediction, args.text)
    
    # Create visualization
    plot_save_path = os.path.join(args.output_dir, f'logit_lens_{args.model_type}.png')
    plot_logit_lens(predictions, final_prediction, save_path=plot_save_path)
    
    # Save raw data
    import json
    data_save_path = os.path.join(args.output_dir, f'logit_lens_data_{args.model_type}.json')
    with open(data_save_path, 'w') as f:
        json.dump({
            'text': args.text,
            'model_type': args.model_type,
            'predictions': predictions,
            'final_prediction': final_prediction
        }, f, indent=2)
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()