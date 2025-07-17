#!/usr/bin/env python3
"""
Run inference with hooks on a saved checkpoint.
Matrix visualization for attention patterns.

Run with: python -m run_inference_with_hooks [args]
Or set PYTHONPATH=/path/to/repo before running
"""

from collections import defaultdict
from matplotlib import colors
import numpy as np
import torch
import argparse
import os
import sys

from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import (
    AttentionExtractionHook,
    FFNActivationTracker
)
from src.mytokenizers import create_tokenizer, from_pretrained
import logging
import json
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device, arg_model_type):
    """Load model from checkpoint."""
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


def create_attention_matrices_visualization(attention_hook, output_dir=None, max_layers=3, max_heads=4, exclude_first_n=3):
    """
    Create simple matrix visualizations of attention patterns.
    
    Args:
        attention_hook: Hook containing attention data
        output_dir: Directory to save visualizations
        max_layers: Maximum layers to show
        max_heads: Maximum heads per layer
        exclude_first_n: Number of initial tokens to exclude from visualization
    """
    logger.info("Creating attention matrix visualizations (final sequence only)...")
    
    if not attention_hook.attention_data:
        logger.warning("No attention data available for matrix visualization")
        return
    
    # find the maximum position (final generation step)
    max_position = max(record['position'] for record in attention_hook.attention_data)
    logger.info(f"Analyzing final sequence at position {max_position}")
    
    # group attention data by layer and head 
    attention_by_layer_head = defaultdict(list)
    for record in attention_hook.attention_data:
        if record['position'] == max_position and 'attention_matrix' in record:
            key = (record['layer'], record['head'])
            attention_by_layer_head[key].append(record)
    
    if not attention_by_layer_head:
        logger.warning("No attention matrices found in data")
        return
    
    # select layers and heads to visualize
    layer_head_pairs = sorted(attention_by_layer_head.keys())[:max_layers * max_heads]
    
    # calculate grid dimensions
    n_plots = len(layer_head_pairs)
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = (n_plots + 3) // 6, 6
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (layer, head) in enumerate(layer_head_pairs):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        records = attention_by_layer_head[(layer, head)]
        
        # use the last record (final generation step) for this layer/head
        record = records[-1] if records else records[0]
        attention_matrix = record['attention_matrix'].numpy()
        tokens = record.get('tokens', [])
        
        # EXCLUDE FIRST N TOKENS FROM VISUALIZATION
        if exclude_first_n > 0:
            # Get the sequence length after excluding tokens
            seq_len = attention_matrix.shape[0]
            if seq_len > exclude_first_n:
                # Slice the attention matrix to exclude first N tokens
                # Keep only the part from exclude_first_n onwards
                attention_matrix = attention_matrix[exclude_first_n:, exclude_first_n:]
                display_tokens = tokens[exclude_first_n:] if tokens else [f"pos_{i}" for i in range(exclude_first_n, seq_len)]
                
                logger.info(f"Layer {layer}, Head {head}: Excluded first {exclude_first_n} tokens, showing {attention_matrix.shape[0]} x {attention_matrix.shape[1]} matrix")
            else:
                logger.warning(f"Layer {layer}, Head {head}: Sequence too short ({seq_len}) to exclude {exclude_first_n} tokens")
                display_tokens = tokens if tokens else [f"pos_{i}" for i in range(seq_len)]
        else:
            # No exclusion - use original matrix
            max_seq_len = attention_matrix.shape[0]
            display_tokens = tokens[:max_seq_len] if tokens else [f"pos_{i}" for i in range(max_seq_len)]
        
        # create heatmap
        power = 0.8
        plot_matrix = np.power(attention_matrix, power)
    
        im = ax.imshow(plot_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        #cbar = plt.colorbar(im, ax=ax)
        #cbar.set_label(f'Attention Weight (power={power})', rotation=270, labelpad=15)
    
        # plot_matrix = attention_matrix + 1e-6
        # vmin = plot_matrix[plot_matrix > 0].min()
        # vmax = plot_matrix.max()
        # norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        # im = ax.imshow(plot_matrix, cmap='Blues', aspect='auto', norm=norm)
        #im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=8)
        ax.set_yticklabels(display_tokens, fontsize=8)

        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Update title to indicate exclusion
        if exclude_first_n > 0:
            ax.set_title(f'Layer {layer}, Head {head}\nExcluding first {exclude_first_n} tokens')
        else:
            ax.set_title(f'Layer {layer}, Head {head}\nFinal Sequence (pos {record["position"]})')
        
        ax.grid(True, alpha=0.3)
    
    # hide unused subplots
    for idx in range(len(layer_head_pairs), len(axes)):
        axes[idx].set_visible(False)
    
    title_suffix = f" (excluding first {exclude_first_n} tokens)" if exclude_first_n > 0 else ""
    plt.suptitle(f'Attention Weight Matrices{title_suffix}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'attention_matrices_exclude_{exclude_first_n}.png' if exclude_first_n > 0 else 'attention_matrices.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        logger.info(f"Attention matrices saved to {output_dir}/{filename}")
    
    plt.show()

def analyze_attention_patterns(attention_hook, output_file=None):
    """Analyze and optionally save attention patterns."""
    logger.info("\n=== Attention Pattern Analysis ===")
    
    # get unique tokens that received/gave attention
    all_tokens = set()
    for record in attention_hook.attention_data:
        for edge in record['edges']:
            all_tokens.add(edge['source_token'])
            all_tokens.add(edge['target_token'])
    
    logger.info(f"Unique tokens involved in attention: {len(all_tokens)}")
    
    # analyze attention by layer/head
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
    
    # find most attended tokens
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
    
    # save detailed data if wanted
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
    # basic args
    parser = argparse.ArgumentParser(description='Run inference with hooks and visualization')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='vanilla',
                        help='Directory to save visualizations and analysis')
    parser.add_argument('--model-type', type=str, default='vanilla', choices=['vanilla', 'tft', 'symbolic'])
    parser.add_argument('--prompt', type=str, default=" blah blah blah The name of the thing and are All cats are fluffy. Fluffy is a cat. So Fluffy is a") 
    # Ben saw a dog. He smiled. Mia saw a cat. She laughed. Ben saw a dog. Mia saw a cat. She
    # "The door was open. Tim had a key to the door. Tim used", 
    
    parser.add_argument('--max-tokens', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # settings for the functions
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Threshold for attention extraction')
    parser.add_argument('--save-attention', type=str, default=None,
                        help='Path to save attention analysis JSON')
    parser.add_argument('--max-matrix-layers', type=int, default=6,
                        help='Maximum layers to show in matrix visualization')
    parser.add_argument('--max-matrix-heads', type=int, default=6,
                        help='Maximum heads per layer in matrix visualization')
    parser.add_argument('--track-ffn', action='store_true',
                        help='Enable FFN hook tracking')
    
    args = parser.parse_args()
    args.output_dir = os.path.join('outputs', 'inference', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    
    # create tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = from_pretrained(args.tokenizer)
    else:
        tokenizer = create_tokenizer(args.tokenizer)
    
    # create hooks
    hooks = []
    attention_hook = AttentionExtractionHook(
        threshold=args.attention_threshold,
        store_values=False
    )
    hooks.append(attention_hook)
    
    if args.track_ffn:
        ffn_activation_tracker = FFNActivationTracker()
        hooks.append(ffn_activation_tracker)
 
    logger.info(f"Running with {len(hooks)} hooks: {[h.name for h in hooks]}")
    
    # run generation
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
        hooks=hooks
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("GENERATED TEXT:")
    logger.info(f"{'='*60}")
    print(generated_text)
    logger.info(f"{'='*60}\n")
    
    # analyze hooks
    attention_hook = next((h for h in hooks if isinstance(h, AttentionExtractionHook)), None)
    if attention_hook:
        # text analysis
        save_path = os.path.join(args.output_dir, 'attention_analysis.json') if args.save_attention else None
        analyze_attention_patterns(attention_hook, save_path)
        
        # generate matrix visualizations
        logger.info("\n=== Creating Matrix Visualizations ===")
        try:
            create_attention_matrices_visualization(
                attention_hook, 
                args.output_dir,
                max_layers=args.max_matrix_layers,
                max_heads=args.max_matrix_heads,
                exclude_first_n=0
            )
            logger.info(f"Matrix visualizations saved to: {args.output_dir}")
        except Exception as e:
            logger.error(f"Error creating matrix visualizations: {e}")
    
    # report activation stats if tracked
    if args.track_ffn:
        ffn_tracker = next((h for h in hooks if isinstance(h, FFNActivationTracker)), None)
        if ffn_tracker and ffn_tracker.activations:
            logger.info(f"\nTracked {len(ffn_tracker.activations)} FFN activation records")
            avg_input_norm = sum(a['input_norm'] for a in ffn_tracker.activations) / len(ffn_tracker.activations)
            avg_output_norm = sum(a['output_norm'] for a in ffn_tracker.activations) / len(ffn_tracker.activations)
            logger.info(f"Average FFN input norm: {avg_input_norm:.4f}")
            logger.info(f"Average FFN output norm: {avg_output_norm:.4f}")
    
    logger.info(f"\nAnalysis complete! Check {args.output_dir} for saved results.")


if __name__ == "__main__":
    main()