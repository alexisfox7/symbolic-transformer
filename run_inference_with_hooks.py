#!/usr/bin/env python3
"""
Run inference with hooks on a saved checkpoint.
"""

import torch
import argparse
import os
import sys

from model import get_model
from config import TransformerConfig
from inference.generation import run_generation
from inference.hooks import (
    create_attention_extraction_hook,
    AttentionExtractionHook,
    SymbolicStreamHook,
    ActivationHook
)
from mytokenizers import create_tokenizer, from_pretrained
# Checkpoint loading handled inline
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    """Analyze attention patterns from FINAL SEQUENCE ONLY."""
    
    # Filter to only the final generation step
    if not attention_hook.attention_data:
        print("No attention data found")
        return
    
    max_step = max(record.get('position', 0) for record in attention_hook.attention_data)
    
    final_step_data = [
        record for record in attention_hook.attention_data 
        if record.get('position', 0) == max_step
    ]
    
    print(f"Filtered from {len(attention_hook.attention_data)} total records to {len(final_step_data)} final-step records")
    
    if not final_step_data:
        print("No final step data found")
        return
    
    # Get sequence length from final step
    seq_len = final_step_data[0].get('attention_matrix', torch.tensor([[]])).shape[0]
    max_possible_edges = seq_len * (seq_len + 1) // 2
    
    print(f"Final sequence length: {seq_len} tokens")
    print(f"Max possible edges per head (causal): {max_possible_edges}")
    print()
    
    # Analyze layer/head stats using ONLY final step data
    layer_head_stats = {}
    for record in final_step_data:
        key = (record['layer'], record['head'])
        edges = record['edges']
        
        if edges:
            weights = [e['weight'] for e in edges]
            layer_head_stats[key] = {
                'edge_count': len(edges),
                'avg_weight': sum(weights) / len(weights),
                'max_weight': max(weights)
            }
    
    print("Attention statistics by layer/head (FINAL SEQUENCE ONLY):")
    for (layer, head), stats in sorted(layer_head_stats.items()):
        print(f"  Layer {layer}, Head {head}: "
              f"{stats['edge_count']} edges, "
              f"avg weight: {stats['avg_weight']:.4f}, "
              f"max weight: {stats['max_weight']:.4f}")
    
    # Token attention using ONLY final step data
    token_attention = {}
    for record in final_step_data:
        for edge in record['edges']:
            target = edge['target_token']
            if target not in token_attention:
                token_attention[target] = 0
            token_attention[target] += edge['weight']
    
    top_attended = sorted(token_attention.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most attended tokens (FINAL SEQUENCE ONLY):")
    for token, total_weight in top_attended:
        print(f"  '{token}': {total_weight:.4f}")
    
    # Save data if requested
    if output_file:
        import json
        final_data = {
            'sequence_length': seq_len,
            'max_possible_edges': max_possible_edges,
            'layer_head_stats': {f"L{k[0]}_H{k[1]}": v for k, v in layer_head_stats.items()},
            'top_attended_tokens': top_attended,
            'total_records': len(final_step_data)
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"\nFinal sequence analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with hooks on a checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="He liked to eat", 
                        help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=11, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, 
                        help='Top-k sampling')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
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