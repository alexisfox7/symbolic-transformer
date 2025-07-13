#!/usr/bin/env python3
"""
Run inference with hooks on a saved checkpoint.
Matrix visualization for attention patterns.
"""

import torch
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import (
    create_attention_extraction_hook,
    AttentionExtractionHook,
    SymbolicStreamHook,
    ActivationHook
)
from src.mytokenizers import create_tokenizer, from_pretrained
import logging
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, device='cpu', arg_model_type = "vanilla"):
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
    model_type = arg_model_type 
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


def create_attention_matrices_visualization(attention_hook, output_dir=None, max_layers=3, max_heads=4):
    """
    Create simple matrix visualizations of attention patterns.
    Shows actual attention weight matrices for selected layers/heads.
    ONLY analyzes the final complete sequence (last generation step).
    
    Args:
        attention_hook: AttentionExtractionHook with collected data
        output_dir: Directory to save visualizations
        max_layers: Maximum number of layers to visualize
        max_heads: Maximum number of heads per layer to visualize
    """
    logger.info("Creating attention matrix visualizations (final sequence only)...")
    
    if not attention_hook.attention_data:
        logger.warning("No attention data available for matrix visualization")
        return
    
    # Find the maximum position (final generation step)
    max_position = max(record['position'] for record in attention_hook.attention_data)
    logger.info(f"Analyzing final sequence at position {max_position}")
    
    # Group attention data by layer and head - ONLY from final step
    attention_by_layer_head = defaultdict(list)
    for record in attention_hook.attention_data:
        if record['position'] == max_position and 'attention_matrix' in record:
            key = (record['layer'], record['head'])
            attention_by_layer_head[key].append(record)
    
    if not attention_by_layer_head:
        logger.warning("No attention matrices found in data")
        return
    
    # Select layers and heads to visualize
    layer_head_pairs = sorted(attention_by_layer_head.keys())[:max_layers * max_heads]
    
    # Calculate grid dimensions
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
        #layer_head_pairs = layer_head_pairs[:16]  # Limit to 16 plots max
    
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
        
        # Use the last record (final generation step) for this layer/head
        record = records[-1] if records else records[0]
        attention_matrix = record['attention_matrix'].numpy()
        tokens = record.get('tokens', [])
        
        # Limit matrix size for readability
        max_seq_len = attention_matrix.shape[0]
        attention_matrix = attention_matrix[:max_seq_len, :max_seq_len]
        display_tokens = tokens[:max_seq_len] if tokens else [f"pos_{i}" for i in range(max_seq_len)]
        
        # Create heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        # Set ticks and labels
        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize = 8)
        ax.set_yticklabels(display_tokens, fontsize = 8)
        
        # Labels and title
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Layer {layer}, Head {head}\nFinal Sequence (pos {record["position"]})')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Highlight strong attention weights with text annotations
        if attention_matrix.shape[0] <= 10 and attention_matrix.shape[1] <= 10:
            for i in range(attention_matrix.shape[0]):
                for j in range(attention_matrix.shape[1]):
                    weight = attention_matrix[i, j]
                    if weight > 0.1:  # Only show significant weights
                        color = 'white' if weight > 0.5 else 'black'
                        ax.text(j, i, f'{weight:.2f}', ha='center', va='center', 
                               color=color, fontsize=8, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(layer_head_pairs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Attention Weight Matrices', fontsize=16, fontweight='bold', y = 0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'attention_matrices.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Attention matrices saved to {output_dir}/attention_matrices.png")
    
    plt.show()

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

def analyze_embeddings_quick(model, tokenizer):
    """Quick embedding analysis."""
    embeddings = model.transformer.wte.weight.data
    norms = torch.norm(embeddings, dim=1)
    
    print(f"\n=== EMBEDDING ANALYSIS ===")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Norm stats: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}")
    
    # Check for near-zero embeddings
    near_zero = (norms < 1e-6).sum().item()
    print(f"Near-zero embeddings: {near_zero}")
    
    # Check specific tokens
    test_tokens = ["Ben", "She", "the", ".", ","]
    for token in test_tokens:
        try:
            token_id = tokenizer.encode(token, add_special_tokens=False)[0]
            norm = norms[token_id].item()
            print(f"'{token}' (ID {token_id}): norm = {norm:.6f}")
        except:
            pass


def main():
    # basic args
    parser = argparse.ArgumentParser(description='Run inference with hooks and visualization')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='vanilla',
                        help='Directory to save visualizations and analysis')
    parser.add_argument('--model-type', type=str, default='vanilla')
    parser.add_argument('--prompt', type=str, default="Ben saw a dog. He smiled. Mia saw a cat. She laughed. Ben saw a dog. Mia saw a cat. She", 
                        help='Text prompt for generation') # "The door was open. Tim had a key to the door. Tim used", 
    parser.add_argument('--max-tokens', type=int, default=2,  
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, 
                        help='Top-k sampling')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer type (character, gpt2, or path to pretrained)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    
    # settings for the functions
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Threshold for attention extraction')
    parser.add_argument('--save-attention', type=str, default=None,
                        help='Path to save attention analysis JSON')
    parser.add_argument('--track-activations', action='store_true',
                        help='Track FFN activations')
    parser.add_argument('--max-matrix-layers', type=int, default=6,
                        help='Maximum layers to show in matrix visualization')
    parser.add_argument('--max-matrix-heads', type=int, default=6,
                        help='Maximum heads per layer in matrix visualization')
    
    args = parser.parse_args()
    
    # create output dir
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
    
    # Create hooks
    hooks = []
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
        hooks=hooks
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("GENERATED TEXT:")
    logger.info(f"{'='*60}")
    print(generated_text)
    logger.info(f"{'='*60}\n")
    
    # Analyze hooks
    # Analyze attention patterns
    attention_hook = next((h for h in hooks if isinstance(h, AttentionExtractionHook)), None)
    if attention_hook:
        # Text analysis
        save_path = os.path.join(args.output_dir, 'attention_analysis.json') if args.save_attention else None
        analyze_attention_patterns(attention_hook, save_path)
        analyze_embeddings_quick(model, tokenizer)
        
        # Generate matrix visualizations
        logger.info("\n=== Creating Matrix Visualizations ===")
        try:
            create_attention_matrices_visualization(
                attention_hook, 
                args.output_dir,
                max_layers=args.max_matrix_layers,
                max_heads=args.max_matrix_heads
            )
            logger.info(f"Matrix visualizations saved to: {args.output_dir}")
        except Exception as e:
            logger.error(f"Error creating matrix visualizations: {e}")
    
    # Report activation stats if tracked
    if args.track_activations:
        activation_hook = next((h for h in hooks if isinstance(h, ActivationHook)), None)
        if activation_hook and activation_hook.activations:
            logger.info(f"\nTracked {len(activation_hook.activations)} activation records")
            avg_input_norm = sum(a['input_norm'] for a in activation_hook.activations) / len(activation_hook.activations)
            avg_output_norm = sum(a['output_norm'] for a in activation_hook.activations) / len(activation_hook.activations)
            logger.info(f"Average FFN input norm: {avg_input_norm:.4f}")
            logger.info(f"Average FFN output norm: {avg_output_norm:.4f}")
    
    logger.info(f"\nAnalysis complete! Check {args.output_dir} for saved results.")


if __name__ == "__main__":
    main()