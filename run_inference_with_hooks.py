#!/usr/bin/env python3
"""
Run inference with hooks on a saved checkpoint.
"""

import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import (
    create_attention_extraction_hook,
    AttentionExtractionHook,
    SymbolicStreamHook,
    ActivationHook
)
from src.mytokenizers import create_tokenizer, from_pretrained
# Checkpoint loading handled inline
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        # Handle both dict and TransformerConfig object
        if hasattr(config_data, '__dict__'):
            config = config_data  # Already a TransformerConfig
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    # Determine model type from training_result or guess from structure
    model_type = 'vanilla'  # default
    if 'training_result' in checkpoint and 'model_type' in checkpoint['training_result']:
        model_type = checkpoint['training_result']['model_type']
    elif 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # Try to detect from model structure
        state_dict = checkpoint.get('model_state_dict', {})
        if any('alibi' in k.lower() or 'vocab_ffn' in k.lower() for k in state_dict.keys()):
            model_type = 'symbolic'
    
    logger.info(f"Model type: {model_type}")
    
    # Create model
    model = get_model(model_type, config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading directly if it's just the state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
    
    return model, config


def analyze_attention_patterns(attention_hook, output_file=None):
    """Analyze and optionally save attention patterns."""
    logger.info("\n=== Attention Pattern Analysis ===")
    
    # Get unique tokens that received/gave attention
    all_tokens = set()
    for record in attention_hook.attention_data:
        for edge in record['edges']:
            all_tokens.add(edge['source_token'])
            all_tokens.add(edge['target_token'])
    
    logger.info(f"Unique tokens involved in attention: {len(all_tokens)}")
    
    # Analyze attention by layer/head
    layer_head_stats = {}
    for record in attention_hook.attention_data:
        key = (record['layer'], record['head'])
        if key not in layer_head_stats:
            layer_head_stats[key] = {
                'edge_count': 0,
                'avg_weight': 0,
                'max_weight': 0
            }
        
        stats = layer_head_stats[key]
        edges = record['edges']
        if edges:
            weights = [e['weight'] for e in edges]
            stats['edge_count'] += len(edges)
            stats['avg_weight'] = sum(weights) / len(weights)
            stats['max_weight'] = max(weights)
    
    logger.info("\nAttention statistics by layer/head:")
    for (layer, head), stats in sorted(layer_head_stats.items()):
        logger.info(f"  Layer {layer}, Head {head}: "
                   f"{stats['edge_count']} edges, "
                   f"avg weight: {stats['avg_weight']:.4f}, "
                   f"max weight: {stats['max_weight']:.4f}")
    
    # Find most attended tokens
    token_attention = {}
    for record in attention_hook.attention_data:
        for edge in record['edges']:
            target = edge['target_token']
            if target not in token_attention:
                token_attention[target] = 0
            token_attention[target] += edge['weight']
    
    top_attended = sorted(token_attention.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("\nTop 10 most attended tokens:")
    for token, total_weight in top_attended:
        logger.info(f"  '{token}': {total_weight:.4f}")
    
    # Save detailed data if requested
    if output_file:
        data_to_save = {
            'attention_records': len(attention_hook.attention_data),
            'unique_tokens': list(all_tokens),
            'layer_head_stats': {f"L{k[0]}_H{k[1]}": v for k, v in layer_head_stats.items()},
            'top_attended_tokens': top_attended,
            'full_data': attention_hook.attention_data[:10]  # Save first 10 records as example
        }
        
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        logger.info(f"\nDetailed attention data saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with hooks on a checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time", 
                        help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, 
                        help='Top-k sampling')
    parser.add_argument('--tokenizer', type=str, default='character',
                        help='Tokenizer type (character, gpt2, or path to pretrained)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Threshold for attention extraction')
    parser.add_argument('--save-attention', type=str, default=None,
                        help='Path to save attention analysis JSON')
    parser.add_argument('--track-activations', action='store_true',
                        help='Track FFN activations')
    parser.add_argument('--no-hooks', action='store_true',
                        help='Run without hooks for comparison')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = from_pretrained(args.tokenizer)
    else:
        tokenizer = create_tokenizer(args.tokenizer)
    
    # Create hooks
    hooks = []
    if not args.no_hooks:
        # Always add attention extraction
        attention_hook = create_attention_extraction_hook(
            threshold=args.attention_threshold,
            store_values=False
        )
        hooks.append(attention_hook)
        
        # Optionally add activation tracking
        if args.track_activations:
            activation_hook = ActivationHook()
            hooks.append(activation_hook)
        
        # Add symbolic stream tracker if it's a symbolic model
        if hasattr(model, 'transformer') and hasattr(model.transformer.h[0], 'ffn'):
            if hasattr(model.transformer.h[0].ffn, 'vocab_embeddings_ref'):
                symbolic_hook = SymbolicStreamHook()
                hooks.append(symbolic_hook)
                logger.info("Detected symbolic model, adding stream tracker")
        
        logger.info(f"Running with {len(hooks)} hooks: {[h.name for h in hooks]}")
    else:
        logger.info("Running without hooks")
    
    # Run generation
    logger.info(f"\nGenerating text from prompt: '{args.prompt}'")
    logger.info(f"Parameters: max_tokens={args.max_tokens}, temp={args.temperature}, top_k={args.top_k}")
    
    ids, generated_text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        show_progress=True,
        hooks=hooks if not args.no_hooks else None
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("GENERATED TEXT:")
    logger.info(f"{'='*60}")
    print(generated_text)
    logger.info(f"{'='*60}\n")
    
    # Analyze hooks if used
    if not args.no_hooks and hooks:
        # Analyze attention patterns
        attention_hook = next((h for h in hooks if isinstance(h, AttentionExtractionHook)), None)
        if attention_hook:
            analyze_attention_patterns(attention_hook, args.save_attention)
        
        # Report activation stats if tracked
        if args.track_activations:
            activation_hook = next((h for h in hooks if isinstance(h, ActivationHook)), None)
            if activation_hook and activation_hook.activations:
                logger.info(f"\nTracked {len(activation_hook.activations)} activation records")
                avg_input_norm = sum(a['input_norm'] for a in activation_hook.activations) / len(activation_hook.activations)
                avg_output_norm = sum(a['output_norm'] for a in activation_hook.activations) / len(activation_hook.activations)
                logger.info(f"Average FFN input norm: {avg_input_norm:.4f}")
                logger.info(f"Average FFN output norm: {avg_output_norm:.4f}")


if __name__ == "__main__":
    main()