#!/usr/bin/env python3
"""
Analyze Kronecker matrices in Symbolic/TFT models to understand head mixing patterns.

The Kronecker structure creates head-to-head interactions through:
- v_tmp: Controls how heads mix in value computation 
- proj_tmp: Controls how heads mix in output projection

Usage:
    python -m kronecker_analyzer --checkpoint path/to/checkpoint.pt --model-type symbolic
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from collections import defaultdict
import json

from src.model import get_model
from src.config import TransformerConfig
from src.mytokenizers import create_tokenizer, from_pretrained
from src.inference.generation import run_generation
from src.hooks.base import InferenceHook, HookManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KroneckerAnalysisHook(InferenceHook):
    """Hook to analyze Kronecker matrix behavior during inference."""
    
    def __init__(self, model):
        super().__init__("kronecker_analysis")
        self.model = model
        self.kronecker_data = []
        
    def on_generation_begin(self, prompt_tokens, state):
        """Extract and analyze Kronecker matrices from the model."""
        logger.info("Analyzing Kronecker matrices in model...")
        
        # Find layers with Kronecker structure
        for layer_idx, layer in enumerate(self.model.transformer.h):
            if not hasattr(layer, 'attn'):
                continue
                
            attn = layer.attn
            layer_analysis = {
                'layer': layer_idx,
                'has_v_kronecker': False,
                'has_proj_kronecker': False
            }
            
            # Analyze V matrix Kronecker structure
            if hasattr(attn, 'use_v') and attn.use_v == 'kronecker' and hasattr(attn, 'v_tmp'):
                v_matrix = attn.v_tmp.detach().cpu()
                layer_analysis.update(self._analyze_kronecker_matrix(v_matrix, 'v_tmp'))
                layer_analysis['has_v_kronecker'] = True
                layer_analysis['v_matrix'] = v_matrix.numpy()
                
            # Analyze projection Kronecker structure  
            if hasattr(attn, 'use_proj') and attn.use_proj == 'kronecker' and hasattr(attn, 'proj_tmp'):
                proj_matrix = attn.proj_tmp.detach().cpu()
                layer_analysis.update(self._analyze_kronecker_matrix(proj_matrix, 'proj_tmp', prefix='proj_'))
                layer_analysis['has_proj_kronecker'] = True
                layer_analysis['proj_matrix'] = proj_matrix.numpy()
            
            self.kronecker_data.append(layer_analysis)
    
    def _analyze_kronecker_matrix(self, matrix, name, prefix=''):
        """Analyze properties of a Kronecker matrix."""
        analysis = {}
        
        # Basic properties
        analysis[f'{prefix}matrix_shape'] = matrix.shape
        analysis[f'{prefix}matrix_norm'] = matrix.norm().item()
        analysis[f'{prefix}matrix_rank'] = torch.linalg.matrix_rank(matrix).item()
        analysis[f'{prefix}matrix_condition'] = torch.linalg.cond(matrix).item()
        
        # Spectral analysis
        eigenvals = torch.linalg.eigvals(matrix)
        analysis[f'{prefix}eigenvalues_real'] = eigenvals.real.numpy().tolist()
        analysis[f'{prefix}eigenvalues_imag'] = eigenvals.imag.numpy().tolist()
        analysis[f'{prefix}spectral_radius'] = torch.max(torch.abs(eigenvals)).item()
        
        # Mixing analysis - how much each head contributes to others
        # Row sums show how much each head sends to others
        row_sums = matrix.sum(dim=1)
        analysis[f'{prefix}row_sums'] = row_sums.numpy().tolist()
        analysis[f'{prefix}row_entropy'] = self._compute_entropy(F.softmax(matrix, dim=1)).numpy().tolist()
        
        # Column sums show how much each head receives from others  
        col_sums = matrix.sum(dim=0)
        analysis[f'{prefix}col_sums'] = col_sums.numpy().tolist()
        analysis[f'{prefix}col_entropy'] = self._compute_entropy(F.softmax(matrix, dim=0)).numpy().tolist()
        
        # Diagonal dominance - how much heads interact with themselves vs others
        diagonal = torch.diag(matrix)
        off_diagonal_sum = matrix.sum() - diagonal.sum()
        analysis[f'{prefix}diagonal_sum'] = diagonal.sum().item()
        analysis[f'{prefix}off_diagonal_sum'] = off_diagonal_sum.item()
        analysis[f'{prefix}diagonal_dominance'] = (diagonal.sum() / matrix.sum()).item()
        
        # Head specialization - variance in row/column patterns
        analysis[f'{prefix}row_variance'] = matrix.var(dim=1).mean().item()
        analysis[f'{prefix}col_variance'] = matrix.var(dim=0).mean().item()
        
        return analysis
    
    def _compute_entropy(self, prob_matrix):
        """Compute entropy of probability distributions (rows/cols)."""
        # Add small epsilon to avoid log(0)
        prob_matrix = prob_matrix + 1e-10
        return -(prob_matrix * torch.log(prob_matrix)).sum(dim=1)


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


def visualize_kronecker_matrices(kronecker_data, output_dir):
    """Create comprehensive visualizations of Kronecker matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter layers that have Kronecker matrices
    v_layers = [d for d in kronecker_data if d['has_v_kronecker']]
    proj_layers = [d for d in kronecker_data if d['has_proj_kronecker']]
    
    if not v_layers and not proj_layers:
        logger.warning("No Kronecker matrices found in model")
        return
    
    # 1. Matrix Heatmaps
    if v_layers:
        plot_matrix_heatmaps(v_layers, 'v_matrix', 'V Matrix Kronecker', 
                           os.path.join(output_dir, 'v_kronecker_heatmaps.png'))
    
    if proj_layers:
        plot_matrix_heatmaps(proj_layers, 'proj_matrix', 'Projection Matrix Kronecker',
                           os.path.join(output_dir, 'proj_kronecker_heatmaps.png'))
    
    # 2. Mixing Analysis
    plot_mixing_analysis(kronecker_data, output_dir)
    
    # 3. Spectral Analysis
    plot_spectral_analysis(kronecker_data, output_dir)
    
    # 4. Head Interaction Patterns
    plot_head_interactions(kronecker_data, output_dir)


def plot_matrix_heatmaps(layer_data, matrix_key, title, save_path):
    """Plot heatmaps of Kronecker matrices across layers."""
    n_layers = len(layer_data)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, layer_data_item in enumerate(layer_data):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        matrix = layer_data_item[matrix_key]
        layer_idx = layer_data_item['layer']
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdBu_r', center=0, aspect='auto')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Head (To)')
        ax.set_ylabel('Head (From)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add value annotations for small matrices
        if matrix.shape[0] <= 8:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    ax.text(col, row, f'{matrix[row, col]:.2f}',
                           ha='center', va='center', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(layer_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title} - Head Mixing Patterns', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Matrix heatmaps saved to: {save_path}")
    plt.show()


def plot_mixing_analysis(kronecker_data, output_dir):
    """Analyze and plot mixing patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    layers = [d['layer'] for d in kronecker_data]
    v_diagonal_dom = [d.get('diagonal_dominance', 0) for d in kronecker_data]
    proj_diagonal_dom = [d.get('proj_diagonal_dominance', 0) for d in kronecker_data]
    v_entropy = [np.mean(d.get('row_entropy', [0])) for d in kronecker_data]
    proj_entropy = [np.mean(d.get('proj_row_entropy', [0])) for d in kronecker_data]
    
    # 1. Diagonal Dominance
    v_layers = [i for i, d in enumerate(kronecker_data) if d['has_v_kronecker']]
    proj_layers = [i for i, d in enumerate(kronecker_data) if d['has_proj_kronecker']]
    
    if v_layers:
        axes[0, 0].plot([layers[i] for i in v_layers], [v_diagonal_dom[i] for i in v_layers], 
                       'bo-', label='V Matrix', linewidth=2, markersize=6)
    if proj_layers:
        axes[0, 0].plot([layers[i] for i in proj_layers], [proj_diagonal_dom[i] for i in proj_layers], 
                       'ro-', label='Proj Matrix', linewidth=2, markersize=6)
    
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Diagonal Dominance')
    axes[0, 0].set_title('Head Self-Interaction vs Cross-Interaction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Equal mix')
    
    # 2. Mixing Entropy
    if v_layers:
        axes[0, 1].plot([layers[i] for i in v_layers], [v_entropy[i] for i in v_layers], 
                       'bo-', label='V Matrix', linewidth=2, markersize=6)
    if proj_layers:
        axes[0, 1].plot([layers[i] for i in proj_layers], [proj_entropy[i] for i in proj_layers], 
                       'ro-', label='Proj Matrix', linewidth=2, markersize=6)
    
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Average Row Entropy')
    axes[0, 1].set_title('Head Mixing Complexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Matrix Norms
    v_norms = [d.get('matrix_norm', 0) for d in kronecker_data]
    proj_norms = [d.get('proj_matrix_norm', 0) for d in kronecker_data]
    
    if v_layers:
        axes[1, 0].plot([layers[i] for i in v_layers], [v_norms[i] for i in v_layers], 
                       'bo-', label='V Matrix', linewidth=2, markersize=6)
    if proj_layers:
        axes[1, 0].plot([layers[i] for i in proj_layers], [proj_norms[i] for i in proj_layers], 
                       'ro-', label='Proj Matrix', linewidth=2, markersize=6)
    
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Matrix Norm')
    axes[1, 0].set_title('Kronecker Matrix Magnitudes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Condition Numbers (stability)
    v_cond = [d.get('matrix_condition', 1) for d in kronecker_data]
    proj_cond = [d.get('proj_matrix_condition', 1) for d in kronecker_data]
    
    if v_layers:
        axes[1, 1].semilogy([layers[i] for i in v_layers], [v_cond[i] for i in v_layers], 
                           'bo-', label='V Matrix', linewidth=2, markersize=6)
    if proj_layers:
        axes[1, 1].semilogy([layers[i] for i in proj_layers], [proj_cond[i] for i in proj_layers], 
                           'ro-', label='Proj Matrix', linewidth=2, markersize=6)
    
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Condition Number (log scale)')
    axes[1, 1].set_title('Matrix Stability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Warning')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'kronecker_mixing_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Mixing analysis saved to: {save_path}")
    plt.show()


def plot_spectral_analysis(kronecker_data, output_dir):
    """Analyze eigenvalues and spectral properties."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, layer_data in enumerate(kronecker_data):
        layer_idx = layer_data['layer']
        
        # V matrix eigenvalues
        if layer_data['has_v_kronecker']:
            v_eigs_real = np.array(layer_data['eigenvalues_real'])
            v_eigs_imag = np.array(layer_data['eigenvalues_imag'])
            
            axes[0, 0].scatter(v_eigs_real, v_eigs_imag, alpha=0.7, 
                             label=f'Layer {layer_idx}', s=50)
        
        # Proj matrix eigenvalues
        if layer_data['has_proj_kronecker']:
            proj_eigs_real = np.array(layer_data['proj_eigenvalues_real'])
            proj_eigs_imag = np.array(layer_data['proj_eigenvalues_imag'])
            
            axes[0, 1].scatter(proj_eigs_real, proj_eigs_imag, alpha=0.7, 
                             label=f'Layer {layer_idx}', s=50)
    
    # Format eigenvalue plots
    axes[0, 0].set_xlabel('Real Part')
    axes[0, 0].set_ylabel('Imaginary Part')
    axes[0, 0].set_title('V Matrix Eigenvalues')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[0, 1].set_xlabel('Real Part')
    axes[0, 1].set_ylabel('Imaginary Part')
    axes[0, 1].set_title('Projection Matrix Eigenvalues')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Spectral radius progression
    layers = [d['layer'] for d in kronecker_data]
    v_spectral = [d.get('spectral_radius', 0) for d in kronecker_data if d['has_v_kronecker']]
    proj_spectral = [d.get('proj_spectral_radius', 0) for d in kronecker_data if d['has_proj_kronecker']]
    
    v_layer_nums = [d['layer'] for d in kronecker_data if d['has_v_kronecker']]
    proj_layer_nums = [d['layer'] for d in kronecker_data if d['has_proj_kronecker']]
    
    if v_spectral:
        axes[1, 0].plot(v_layer_nums, v_spectral, 'bo-', linewidth=2, markersize=6, label='V Matrix')
    if proj_spectral:
        axes[1, 0].plot(proj_layer_nums, proj_spectral, 'ro-', linewidth=2, markersize=6, label='Proj Matrix')
    
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Spectral Radius')
    axes[1, 0].set_title('Maximum Eigenvalue Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Stability threshold')
    
    # Matrix ranks
    v_ranks = [d.get('matrix_rank', 0) for d in kronecker_data if d['has_v_kronecker']]
    proj_ranks = [d.get('proj_matrix_rank', 0) for d in kronecker_data if d['has_proj_kronecker']]
    
    if v_ranks:
        axes[1, 1].plot(v_layer_nums, v_ranks, 'bo-', linewidth=2, markersize=6, label='V Matrix')
    if proj_ranks:
        axes[1, 1].plot(proj_layer_nums, proj_ranks, 'ro-', linewidth=2, markersize=6, label='Proj Matrix')
    
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Matrix Rank')
    axes[1, 1].set_title('Effective Dimensionality')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'kronecker_spectral_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Spectral analysis saved to: {save_path}")
    plt.show()


def plot_head_interactions(kronecker_data, output_dir):
    """Visualize which heads interact most strongly."""
    # Find the layer with strongest mixing (highest off-diagonal sum)
    best_v_layer = None
    best_proj_layer = None
    max_v_mixing = 0
    max_proj_mixing = 0
    
    for layer_data in kronecker_data:
        if layer_data['has_v_kronecker']:
            off_diag = layer_data.get('off_diagonal_sum', 0)
            if off_diag > max_v_mixing:
                max_v_mixing = off_diag
                best_v_layer = layer_data
                
        if layer_data['has_proj_kronecker']:
            off_diag = layer_data.get('proj_off_diagonal_sum', 0)
            if off_diag > max_proj_mixing:
                max_proj_mixing = off_diag
                best_proj_layer = layer_data
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # V matrix interactions
    if best_v_layer:
        v_matrix = best_v_layer['v_matrix']
        layer_idx = best_v_layer['layer']
        
        # Create network-style visualization
        n_heads = v_matrix.shape[0]
        
        # Plot as directed graph
        im1 = axes[0].imshow(v_matrix, cmap='RdBu_r', center=0)
        axes[0].set_title(f'V Matrix Head Interactions (Layer {layer_idx})')
        axes[0].set_xlabel('To Head')
        axes[0].set_ylabel('From Head')
        
        # Add value annotations
        for i in range(n_heads):
            for j in range(n_heads):
                if abs(v_matrix[i, j]) > 0.1:  # Only show significant values
                    axes[0].text(j, i, f'{v_matrix[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(v_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Proj matrix interactions
    if best_proj_layer:
        proj_matrix = best_proj_layer['proj_matrix']
        layer_idx = best_proj_layer['layer']
        
        im2 = axes[1].imshow(proj_matrix, cmap='RdBu_r', center=0)
        axes[1].set_title(f'Projection Matrix Head Interactions (Layer {layer_idx})')
        axes[1].set_xlabel('To Head')
        axes[1].set_ylabel('From Head')
        
        # Add value annotations
        n_heads = proj_matrix.shape[0]
        for i in range(n_heads):
            for j in range(n_heads):
                if abs(proj_matrix[i, j]) > 0.1:
                    axes[1].text(j, i, f'{proj_matrix[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(proj_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'head_interaction_patterns.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Head interaction patterns saved to: {save_path}")
    plt.show()


def analyze_kronecker_patterns(kronecker_data, output_file=None):
    """Analyze and report Kronecker mixing patterns."""
    logger.info("\n=== KRONECKER MATRIX ANALYSIS ===")
    
    v_layers = [d for d in kronecker_data if d['has_v_kronecker']]
    proj_layers = [d for d in kronecker_data if d['has_proj_kronecker']]
    
    logger.info(f"Found {len(v_layers)} layers with V Kronecker matrices")
    logger.info(f"Found {len(proj_layers)} layers with Projection Kronecker matrices")
    
    # Analyze V matrices
    if v_layers:
        logger.info("\n📊 V Matrix Analysis:")
        for layer_data in v_layers:
            layer = layer_data['layer']
            diag_dom = layer_data.get('diagonal_dominance', 0)
            spectral_radius = layer_data.get('spectral_radius', 0)
            matrix_rank = layer_data.get('matrix_rank', 0)
            
            logger.info(f"  Layer {layer}: Diagonal dominance={diag_dom:.3f}, "
                       f"Spectral radius={spectral_radius:.3f}, Rank={matrix_rank}")
            
            # Identify strongest head interactions
            matrix = layer_data['v_matrix']
            n_heads = matrix.shape[0]
            
            # Find strongest off-diagonal elements
            strongest_interactions = []
            for i in range(n_heads):
                for j in range(n_heads):
                    if i != j and abs(matrix[i, j]) > 0.1:
                        strongest_interactions.append((i, j, matrix[i, j]))
            
            strongest_interactions.sort(key=lambda x: abs(x[2]), reverse=True)
            if strongest_interactions:
                logger.info(f"    Strongest head interactions:")
                for from_head, to_head, weight in strongest_interactions[:3]:
                    logger.info(f"      Head {from_head} → Head {to_head}: {weight:.3f}")
    
    # Analyze Projection matrices
    if proj_layers:
        logger.info("\n📊 Projection Matrix Analysis:")
        for layer_data in proj_layers:
            layer = layer_data['layer']
            diag_dom = layer_data.get('proj_diagonal_dominance', 0)
            spectral_radius = layer_data.get('proj_spectral_radius', 0)
            matrix_rank = layer_data.get('proj_matrix_rank', 0)
            
            logger.info(f"  Layer {layer}: Diagonal dominance={diag_dom:.3f}, "
                       f"Spectral radius={spectral_radius:.3f}, Rank={matrix_rank}")
    
    # Overall patterns
    logger.info("\n🔍 Overall Patterns:")
    
    if v_layers:
        avg_v_diag = np.mean([d.get('diagonal_dominance', 0) for d in v_layers])
        logger.info(f"  Average V matrix diagonal dominance: {avg_v_diag:.3f}")
        if avg_v_diag > 0.7:
            logger.info("    → Heads mostly operate independently")
        elif avg_v_diag > 0.4:
            logger.info("    → Moderate head mixing")
        else:
            logger.info("    → Strong head interactions")
    
    if proj_layers:
        avg_proj_diag = np.mean([d.get('proj_diagonal_dominance', 0) for d in proj_layers])
        logger.info(f"  Average Projection matrix diagonal dominance: {avg_proj_diag:.3f}")
    
    # Save detailed analysis
    if output_file:
        analysis_data = {
            'summary': {
                'v_layers_count': len(v_layers),
                'proj_layers_count': len(proj_layers),
                'avg_v_diagonal_dominance': np.mean([d.get('diagonal_dominance', 0) for d in v_layers]) if v_layers else 0,
                'avg_proj_diagonal_dominance': np.mean([d.get('proj_diagonal_dominance', 0) for d in proj_layers]) if proj_layers else 0
            },
            'detailed_analysis': kronecker_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        logger.info(f"\nDetailed analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Kronecker matrices in Symbolic/TFT transformers')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='symbolic',
                       choices=['symbolic', 'tft'], help='Model type')
    parser.add_argument('--output-dir', type=str, default='kronecker_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--prompt', type=str, default="The cat sat on the mat. The quick brown fox jumps.",
                       help='Text prompt for generation analysis')
    parser.add_argument('--max-tokens', type=int, default=5,
                       help='Maximum tokens to generate')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-analysis', action='store_true',
                       help='Save detailed analysis to JSON file')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join('outputs', 'analysis', args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    
    # Create tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = from_pretrained(args.tokenizer)
    else:
        tokenizer = create_tokenizer(args.tokenizer)
    
    # Create Kronecker analysis hook
    kronecker_hook = KroneckerAnalysisHook(model)
    
    logger.info(f"Analyzing Kronecker matrices in {args.model_type} model")
    logger.info(f"Running generation to trigger analysis: '{args.prompt}'")
    
    # Run generation to trigger analysis
    ids, generated_text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        device=device,
        max_new_tokens=args.max_tokens,
        hooks=[kronecker_hook]
    )
    
    logger.info(f"Generated: {generated_text}")
    
    # Analyze results
    if kronecker_hook.kronecker_data:
        analyze_kronecker_patterns(
            kronecker_hook.kronecker_data,
            os.path.join(output_dir, 'kronecker_analysis.json') if args.save_analysis else None
        )
        
        # Create visualizations
        try:
            visualize_kronecker_matrices(kronecker_hook.kronecker_data, output_dir)
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"\nKronecker analysis complete! Results saved to: {output_dir}")
    else:
        logger.warning("No Kronecker matrices found in model. Check that model uses 'kronecker' for use_v or use_proj.")


if __name__ == "__main__":
    main()