#!/usr/bin/env python3
"""
Run inference with hooks on a saved checkpoint.
Enhanced with graph visualization capabilities for attention patterns and reasoning flows.
Now includes simple matrix visualization for attention patterns.
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


# def create_attention_pattern_summary(attention_hook, output_dir=None):
#     """
#     Create a summary visualization showing attention patterns across all layers and heads.
#     Simple bar charts and statistics.
#     """
#     logger.info("Creating attention pattern summary...")
    
#     if not attention_hook.attention_data:
#         logger.warning("No attention data for pattern summary")
#         return
    
#     # Collect statistics
#     layer_stats = defaultdict(lambda: {'total_weight': 0, 'max_weight': 0, 'edge_count': 0})
#     head_stats = defaultdict(lambda: {'total_weight': 0, 'max_weight': 0, 'edge_count': 0})
    
#     for record in attention_hook.attention_data:
#         layer = record['layer']
#         head = record['head']
#         edges = record.get('edges', [])
        
#         if edges:
#             weights = [e['weight'] for e in edges]
#             total_weight = sum(weights)
#             max_weight = max(weights)
#             edge_count = len(edges)
            
#             # Update layer stats
#             layer_stats[layer]['total_weight'] += total_weight
#             layer_stats[layer]['max_weight'] = max(layer_stats[layer]['max_weight'], max_weight)
#             layer_stats[layer]['edge_count'] += edge_count
            
#             # Update head stats  
#             head_stats[head]['total_weight'] += total_weight
#             head_stats[head]['max_weight'] = max(head_stats[head]['max_weight'], max_weight)
#             head_stats[head]['edge_count'] += edge_count
    
#     # Create summary plots
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Layer-wise total attention
#     layers = sorted(layer_stats.keys())
#     layer_totals = [layer_stats[l]['total_weight'] for l in layers]
    
#     axes[0, 0].bar(layers, layer_totals, color='skyblue', alpha=0.7)
#     axes[0, 0].set_title('Total Attention Weight by Layer')
#     axes[0, 0].set_xlabel('Layer')
#     axes[0, 0].set_ylabel('Total Attention Weight')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Head-wise total attention
#     heads = sorted(head_stats.keys())
#     head_totals = [head_stats[h]['total_weight'] for h in heads]
    
#     axes[0, 1].bar(heads, head_totals, color='lightcoral', alpha=0.7)
#     axes[0, 1].set_title('Total Attention Weight by Head')
#     axes[0, 1].set_xlabel('Head')
#     axes[0, 1].set_ylabel('Total Attention Weight')
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # Layer-wise edge count
#     layer_edges = [layer_stats[l]['edge_count'] for l in layers]
    
#     axes[1, 0].bar(layers, layer_edges, color='lightgreen', alpha=0.7)
#     axes[1, 0].set_title('Attention Edge Count by Layer')
#     axes[1, 0].set_xlabel('Layer')
#     axes[1, 0].set_ylabel('Number of Attention Edges')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # Head-wise edge count
#     head_edges = [head_stats[h]['edge_count'] for h in heads]
    
#     axes[1, 1].bar(heads, head_edges, color='gold', alpha=0.7)
#     axes[1, 1].set_title('Attention Edge Count by Head')
#     axes[1, 1].set_xlabel('Head')
#     axes[1, 1].set_ylabel('Number of Attention Edges')
#     axes[1, 1].grid(True, alpha=0.3)
    
#     plt.suptitle('Attention Pattern Summary Statistics', fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, 'attention_summary.png'), dpi=300, bbox_inches='tight')
#         logger.info(f"Attention summary saved to {output_dir}/attention_summary.png")
    
#     plt.show()


# def create_attention_graph(attention_hook, output_dir=None, min_weight=0.15, max_nodes=50):
#     """
#     Create and visualize attention graphs from hook data.
    
#     Args:
#         attention_hook: AttentionExtractionHook with collected data
#         output_dir: Directory to save visualizations
#         min_weight: Minimum attention weight to include in graph
#         max_nodes: Maximum number of nodes to include
#     """
#     logger.info("Creating attention graphs...")
    
#     if not attention_hook.attention_data:
#         logger.warning("No attention data available for graph creation")
#         return
    
#     # Aggregate attention patterns across all layers/heads
#     token_attention = defaultdict(float)
#     edge_weights = defaultdict(float)
#     token_positions = {}
    
#     for record in attention_hook.attention_data:
#         for edge in record['edges']:
#             if edge['weight'] >= min_weight:
#                 source_token = edge['source_token']
#                 target_token = edge['target_token']
#                 weight = edge['weight']
                
#                 # Track token importance
#                 token_attention[source_token] += weight
#                 token_attention[target_token] += weight
                
#                 # Track edge weights (aggregate multiple occurrences)
#                 edge_key = (source_token, target_token)
#                 edge_weights[edge_key] += weight
                
#                 # Store positions for layout
#                 token_positions[source_token] = edge['source_pos']
#                 token_positions[target_token] = edge['target_pos']
    
#     # Select most important tokens
#     top_tokens = sorted(token_attention.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
#     selected_tokens = {token for token, _ in top_tokens}
    
#     # Create NetworkX graph
#     G = nx.DiGraph()
    
#     # Add nodes with importance as node attribute
#     for token, importance in top_tokens:
#         G.add_node(token, importance=importance, position=token_positions.get(token, 0))
    
#     # Add edges between selected tokens
#     for (source, target), weight in edge_weights.items():
#         if source in selected_tokens and target in selected_tokens and weight >= min_weight:
#             G.add_edge(source, target, weight=weight)
    
#     if len(G.nodes()) == 0:
#         logger.warning("No nodes in attention graph after filtering")
#         return G
    
#     # Create visualization
#     plt.figure(figsize=(15, 10))
    
#     # Layout based on token positions if available
#     if token_positions:
#         # Use token positions for x-axis, add some y variation
#         pos = {}
#         position_groups = defaultdict(list)
        
#         for token in G.nodes():
#             pos_x = token_positions.get(token, 0)
#             position_groups[pos_x].append(token)
        
#         for pos_x, tokens in position_groups.items():
#             for i, token in enumerate(tokens):
#                 y_offset = (i - len(tokens)/2) * 0.3
#                 pos[token] = (pos_x, y_offset)
#     else:
#         # Fallback to spring layout
#         pos = nx.spring_layout(G, k=3, iterations=50)
    
#     # Node sizes based on importance
#     importance_values = [G.nodes[token]['importance'] for token in G.nodes()]
#     if importance_values:
#         max_importance = max(importance_values)
#         node_sizes = [300 + 1000 * (G.nodes[token]['importance'] / max_importance) for token in G.nodes()]
#     else:
#         node_sizes = [500] * len(G.nodes())
    
#     # Edge widths based on weights
#     edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
#     if edge_weights_list:
#         max_edge_weight = max(edge_weights_list)
#         edge_widths = [1 + 5 * (G[u][v]['weight'] / max_edge_weight) for u, v in G.edges()]
#     else:
#         edge_widths = [1]
    
#     # Draw the graph
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', arrows=True, arrowsize=20)
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
#     # Add edge labels for significant connections
#     strong_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] > np.percentile(edge_weights_list, 75)]
#     edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in strong_edges}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
#     plt.title("Attention Graph: Token Relationships", fontsize=16, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
    
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, 'attention_graph.png'), dpi=300, bbox_inches='tight')
#         logger.info(f"Attention graph saved to {output_dir}/attention_graph.png")
    
#     plt.show()
#     return G


# def create_layer_head_heatmap(attention_hook, output_dir=None):
#     """Create heatmap showing attention patterns across layers and heads."""
#     if not attention_hook.attention_data:
#         logger.warning("No attention data for heatmap")
#         return
    
#     # Collect statistics by layer and head
#     layer_head_stats = defaultdict(lambda: {'edges': 0, 'total_weight': 0, 'max_weight': 0})
    
#     for record in attention_hook.attention_data:
#         key = (record['layer'], record['head'])
#         stats = layer_head_stats[key]
        
#         edges = record['edges']
#         if edges:
#             weights = [e['weight'] for e in edges]
#             stats['edges'] += len(edges)
#             stats['total_weight'] += sum(weights)
#             stats['max_weight'] = max(stats['max_weight'], max(weights))
    
#     if not layer_head_stats:
#         logger.warning("No layer/head statistics available")
#         return
    
#     # Create matrices for visualization
#     layers = sorted(set(k[0] for k in layer_head_stats.keys()))
#     heads = sorted(set(k[1] for k in layer_head_stats.keys()))
    
#     edge_matrix = np.zeros((len(layers), len(heads)))
#     weight_matrix = np.zeros((len(layers), len(heads)))
#     max_weight_matrix = np.zeros((len(layers), len(heads)))
    
#     for (layer, head), stats in layer_head_stats.items():
#         l_idx = layers.index(layer)
#         h_idx = heads.index(head)
#         edge_matrix[l_idx, h_idx] = stats['edges']
#         weight_matrix[l_idx, h_idx] = stats['total_weight']
#         max_weight_matrix[l_idx, h_idx] = stats['max_weight']
    
#     # Create subplot with multiple heatmaps
#     fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
#     # Edge count heatmap
#     sns.heatmap(edge_matrix, annot=True, fmt='.0f', xticklabels=heads, yticklabels=layers, 
#                 ax=axes[0], cmap='Blues', cbar_kws={'label': 'Edge Count'})
#     axes[0].set_title('Attention Edge Count by Layer/Head')
#     axes[0].set_xlabel('Head')
#     axes[0].set_ylabel('Layer')
    
#     # Total weight heatmap
#     sns.heatmap(weight_matrix, annot=True, fmt='.2f', xticklabels=heads, yticklabels=layers, 
#                 ax=axes[1], cmap='Oranges', cbar_kws={'label': 'Total Weight'})
#     axes[1].set_title('Total Attention Weight by Layer/Head')
#     axes[1].set_xlabel('Head')
#     axes[1].set_ylabel('Layer')
    
#     # Max weight heatmap
#     sns.heatmap(max_weight_matrix, annot=True, fmt='.2f', xticklabels=heads, yticklabels=layers, 
#                 ax=axes[2], cmap='Reds', cbar_kws={'label': 'Max Weight'})
#     axes[2].set_title('Maximum Attention Weight by Layer/Head')
#     axes[2].set_xlabel('Head')
#     axes[2].set_ylabel('Layer')
    
#     plt.tight_layout()
    
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(os.path.join(output_dir, 'layer_head_heatmap.png'), dpi=300, bbox_inches='tight')
#         logger.info(f"Layer/head heatmap saved to {output_dir}/layer_head_heatmap.png")
    
#     plt.show()


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
    parser = argparse.ArgumentParser(description='Run inference with hooks and visualization')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='vanilla',
                        help='Directory to save visualizations and analysis')
    parser.add_argument('--model-type', type=str, default='vanilla')
    parser.add_argument('--prompt', type=str, default="Ben saw a dog. He smiled. Mia saw a cat. She laughed. Ben saw a dog. She", 
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
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Threshold for attention extraction')
    parser.add_argument('--save-attention', type=str, default=None,
                        help='Path to save attention analysis JSON')
    parser.add_argument('--track-activations', action='store_true',
                        help='Track FFN activations')
    parser.add_argument('--no-hooks', action='store_true',
                        help='Run without hooks for comparison')
    parser.add_argument('--no-graphs', action='store_true',
                        help='Skip graph generation (faster)')
    parser.add_argument('--no-matrices', action='store_true',
                        help='Skip matrix visualization')
    parser.add_argument('--matrices-only', type=bool, default=True,
                        help='Only generate matrix visualizations (fastest)')
    parser.add_argument('--max-matrix-layers', type=int, default=6,
                        help='Maximum layers to show in matrix visualization')
    parser.add_argument('--max-matrix-heads', type=int, default=6,
                        help='Maximum heads per layer in matrix visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = os.path.join('outputs', 'inference', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    
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
            # Text analysis
            save_path = os.path.join(args.output_dir, 'attention_analysis.json') if args.save_attention else None
            analyze_attention_patterns(attention_hook, save_path)
            
            # Generate visualizations
            if not args.no_graphs and not args.matrices_only:
                logger.info("\n=== Creating Visualizations ===")
                
                try:
                    # Matrix visualizations (simple and fast)
                    if not args.no_matrices:
                        create_attention_matrices_visualization(
                            attention_hook, 
                            args.output_dir,
                            max_layers=args.max_matrix_layers,
                            max_heads=args.max_matrix_heads
                        )
                        
                        #create_attention_pattern_summary(attention_hook, args.output_dir)
                    
                    # More complex visualizations
                    # if not args.matrices_only:
                    #     # Main attention graph
                    #     create_attention_graph(
                    #         attention_hook, 
                    #         args.output_dir, 
                    #         min_weight=0.15,
                    #         max_nodes=50
                    #     )
                        
                    #     # Layer/head heatmap
                    #     create_layer_head_heatmap(attention_hook, args.output_dir)
                    
                    logger.info(f"All visualizations saved to: {args.output_dir}")
                    
                except Exception as e:
                    logger.error(f"Error creating visualizations: {e}")
                    logger.info("Continuing with text analysis only...")
            
            elif args.matrices_only:
                logger.info("\n=== Creating Matrix Visualizations Only ===")
                try:
                    create_attention_matrices_visualization(
                        attention_hook, 
                        args.output_dir,
                        max_layers=args.max_matrix_layers,
                        max_heads=args.max_matrix_heads
                    )
                    
                    #create_attention_pattern_summary(attention_hook, args.output_dir)
                    
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