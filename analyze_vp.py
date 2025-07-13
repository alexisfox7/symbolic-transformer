#!/usr/bin/env python3
"""
Analyze value/projection matrices to detect attention sink behavior.
Enhanced version of run_inference_with_hooks focused on value magnitude patterns.
"""

import torch
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import InferenceHook, InferenceHookManager
from src.mytokenizers import create_tokenizer, from_pretrained
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueSinkAnalysisHook(InferenceHook):
    """
    Comprehensive hook for analyzing value magnitudes and projection patterns
    to detect attention sink behavior (high attention, low values).
    """
    
    def __init__(self, analyze_layers=None, value_threshold=0.1):
        super().__init__("value_sink_analysis")
        self.analyze_layers = analyze_layers or list(range(6))  # First 6 layers
        self.value_threshold = value_threshold
        self.value_data = []
        self.projection_data = []
        self.attention_vs_values = []
        
    def on_attention_computed(self, layer_idx, head_idx, attention_weights, 
                            query, key, value, tokens, position, state):
        """Analyze attention weights vs value magnitudes"""
        
        if layer_idx not in self.analyze_layers:
            return
        
        # Handle tensor shapes properly - attention_weights should be [B, T, T] or [T, T]
        if attention_weights.dim() == 3:
            attention_weights = attention_weights[0]  # Take first batch
        elif attention_weights.dim() == 2:
            pass  # Already correct shape [T, T]
        else:
            logger.warning(f"Unexpected attention_weights shape: {attention_weights.shape}")
            return
            
        # Handle value tensor - should be [B, T, head_dim] or [T, head_dim] 
        if value.dim() == 3:
            value = value[0]  # Take first batch: [T, head_dim]
        elif value.dim() == 2:
            pass  # Already correct shape [T, head_dim]
        else:
            logger.warning(f"Unexpected value shape: {value.shape}")
            return
            
        T, head_dim = value.shape
        if T == 0:
            return
            
        # 1. Compute value magnitudes for each position
        value_norms = torch.norm(value, dim=-1)  # Shape: [T]
        
        # 2. Compute attention received by each position (sum over queries)
        attention_received = attention_weights.sum(dim=0)  # Shape: [T]
        
        # 3. Compute attention given by each position (sum over keys) 
        attention_given = attention_weights.sum(dim=1)  # Shape: [T]
        
        # 4. Store detailed analysis
        analysis_record = {
            'layer': layer_idx,
            'head': head_idx,
            'position_in_generation': position,
            'sequence_length': T,
            'value_norms': value_norms.detach().cpu().numpy().tolist(),
            'attention_received': attention_received.detach().cpu().numpy().tolist(),
            'attention_given': attention_given.detach().cpu().numpy().tolist(),
            'tokens': tokens[:T] if tokens else [f"pos_{i}" for i in range(T)]
        }
        
        # 5. Identify potential sinks (high attention, low values)
        for pos in range(T):
            val_norm = value_norms[pos].detach().cpu().item()
            attn_recv = attention_received[pos].detach().cpu().item()
            attn_given = attention_given[pos].detach().cpu().item()
            
            # Check for sink pattern: high attention received, low value magnitude
            is_potential_sink = (attn_recv > 0.1 and val_norm < self.value_threshold)
            
            sink_record = {
                'layer': layer_idx,
                'head': head_idx,
                'position': pos,
                'generation_step': position,
                'token': tokens[pos] if tokens and pos < len(tokens) else f"pos_{pos}",
                'value_norm': val_norm,
                'attention_received': attn_recv,
                'attention_given': attn_given,
                'is_potential_sink': is_potential_sink,
                'sink_score': attn_recv / (val_norm + 1e-8)  # High score = potential sink
            }
            
            self.attention_vs_values.append(sink_record)
        
        self.value_data.append(analysis_record)
        
        # Print immediate analysis for monitoring
        if head_idx == 0 and position % 5 == 0:  # Print every 5th generation step, head 0 only
            pos_0_val = value_norms[0].detach().cpu().item() if T > 0 else 0
            pos_0_attn = attention_received[0].detach().cpu().item() if T > 0 else 0
            logger.info(f"L{layer_idx}H{head_idx} Step{position}: Pos0 Val:{pos_0_val:.4f} Attn:{pos_0_attn:.4f}")


class ProjectionMatrixAnalysisHook(InferenceHook):
    """
    Hook for analyzing how projection matrices transform token embeddings.
    Focuses on systematic patterns that might create sink behavior.
    """
    
    def __init__(self, model, tokenizer, sample_tokens=100):
        super().__init__("projection_analysis")
        self.model = model
        self.tokenizer = tokenizer
        self.sample_tokens = sample_tokens
        self.projection_analysis = {}
        self.analyzed_layers = set()
        
    def on_generation_begin(self, prompt_tokens, state):
        """Analyze projection matrices once at the start"""
        logger.info("Analyzing projection matrices...")
        
        # Get sample token embeddings
        vocab_size = self.model.config.vocab_size
        sample_indices = torch.randperm(vocab_size)[:self.sample_tokens]
        token_embeddings = self.model.transformer.wte.weight[sample_indices]  # (sample_tokens, n_embd)
        
        for layer_idx, layer in enumerate(self.model.transformer.h):
            if not hasattr(layer, 'attn'):
                continue
                
            attn = layer.attn
            layer_analysis = {'layer': layer_idx}
            
            if hasattr(attn, 'use_v'):
                if attn.use_v == 'normal' and hasattr(attn, 'c_attn'):
                    # Standard V projection
                    self._analyze_standard_v_projection(attn, token_embeddings, layer_analysis)
                elif attn.use_v == 'kronecker' and hasattr(attn, 'v_tmp'):
                    # Kronecker V projection  
                    self._analyze_kronecker_v_projection(attn, token_embeddings, layer_analysis)
                elif attn.use_v == 'none':
                    # Identity V (no projection)
                    layer_analysis.update({
                        'v_type': 'identity',
                        'min_value_norm': torch.norm(token_embeddings, dim=1).min().item(),
                        'max_value_norm': torch.norm(token_embeddings, dim=1).max().item(),
                        'mean_value_norm': torch.norm(token_embeddings, dim=1).mean().item()
                    })
            
            # Analyze output projection if present
            if hasattr(attn, 'use_proj'):
                if attn.use_proj == 'normal' and hasattr(attn, 'c_proj'):
                    self._analyze_output_projection(attn, token_embeddings, layer_analysis)
                elif attn.use_proj == 'kronecker' and hasattr(attn, 'proj_tmp'):
                    self._analyze_kronecker_output_projection(attn, token_embeddings, layer_analysis)
            
            self.projection_analysis[layer_idx] = layer_analysis
            
    def _analyze_standard_v_projection(self, attn, token_embeddings, layer_analysis):
        """Analyze standard V projection matrix"""
        # Extract V weights from concatenated QKV matrix
        full_weight = attn.c_attn.weight  # Shape: (3*n_embd, n_embd)
        n_embd = attn.n_embd
        v_weights = full_weight[2*n_embd:, :]  # V projection part
        
        # Project sample tokens through V
        projected_values = torch.matmul(token_embeddings, v_weights.T)
        value_norms = torch.norm(projected_values, dim=1)
        
        layer_analysis.update({
            'v_type': 'standard',
            'v_matrix_shape': v_weights.shape,
            'v_matrix_norm': torch.norm(v_weights).item(),
            'v_matrix_condition': torch.linalg.cond(v_weights).item(),
            'min_value_norm': value_norms.min().item(),
            'max_value_norm': value_norms.max().item(),
            'mean_value_norm': value_norms.mean().item(),
            'std_value_norm': value_norms.std().item(),
            'zero_like_values': (value_norms < 0.01).sum().item(),  # Count very small values
            'value_norm_percentiles': torch.quantile(value_norms, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])).tolist()
        })
        
    def _analyze_kronecker_v_projection(self, attn, token_embeddings, layer_analysis):
        """Analyze Kronecker-structured V projection"""
        v_matrix = attn._get_kronecker_lifted_tensor(attn.v_tmp)
        projected_values = torch.matmul(token_embeddings, v_matrix)
        value_norms = torch.norm(projected_values, dim=1)
        
        layer_analysis.update({
            'v_type': 'kronecker',
            'v_head_matrix_shape': attn.v_tmp.shape,
            'v_head_matrix_norm': torch.norm(attn.v_tmp).item(),
            'v_head_condition': torch.linalg.cond(attn.v_tmp).item(),
            'full_v_matrix_shape': v_matrix.shape,
            'min_value_norm': value_norms.min().item(),
            'max_value_norm': value_norms.max().item(),
            'mean_value_norm': value_norms.mean().item(),
            'zero_like_values': (value_norms < 0.01).sum().item(),
        })
        
    def _analyze_output_projection(self, attn, token_embeddings, layer_analysis):
        """Analyze standard output projection"""
        if hasattr(attn, 'c_proj'):
            proj_weights = attn.c_proj.weight
            projected = torch.matmul(token_embeddings, proj_weights.T)
            proj_norms = torch.norm(projected, dim=1)
            
            layer_analysis.update({
                'output_proj_type': 'standard',
                'output_proj_condition': torch.linalg.cond(proj_weights).item(),
                'output_proj_min_norm': proj_norms.min().item(),
                'output_proj_max_norm': proj_norms.max().item(),
                'output_proj_mean_norm': proj_norms.mean().item()
            })


def load_model_from_checkpoint(checkpoint_path, device='cpu', arg_model_type="vanilla"):
    """Load model from checkpoint (same as run_inference_with_hooks)"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        if hasattr(config_data, '__dict__'):
            config = config_data
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    model_type = arg_model_type 
    if 'training_result' in checkpoint and 'model_type' in checkpoint['training_result']:
        model_type = checkpoint['training_result']['model_type']
    elif 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    
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


def create_value_sink_visualizations(value_hook, projection_hook, output_dir):
    """Create comprehensive visualizations of value sink patterns"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Attention vs Value Magnitude Scatter Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert data to arrays for plotting
    attention_data = value_hook.attention_vs_values
    if not attention_data:
        logger.warning("No attention vs value data available")
        return
    
    attentions = [d['attention_received'] for d in attention_data]
    values = [d['value_norm'] for d in attention_data]
    layers = [d['layer'] for d in attention_data]
    positions = [d['position'] for d in attention_data]
    
    # Scatter plot: Attention vs Value magnitude
    scatter = axes[0, 0].scatter(attentions, values, c=layers, alpha=0.6, cmap='viridis')
    axes[0, 0].set_xlabel('Attention Received')
    axes[0, 0].set_ylabel('Value Magnitude')
    axes[0, 0].set_title('Attention vs Value Magnitude (Sinks in bottom-right)')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Layer')
    
    # Add diagonal line showing expected relationship
    max_val = max(max(attentions), max(values))
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Expected correlation')
    axes[0, 0].legend()
    
    # 2. Position-wise analysis (focus on first few positions)
    pos_data = defaultdict(lambda: {'attentions': [], 'values': []})
    for d in attention_data:
        if d['position'] <= 10:  # First 10 positions only
            pos_data[d['position']]['attentions'].append(d['attention_received'])
            pos_data[d['position']]['values'].append(d['value_norm'])
    
    positions = sorted(pos_data.keys())
    avg_attentions = [np.mean(pos_data[p]['attentions']) for p in positions]
    avg_values = [np.mean(pos_data[p]['values']) for p in positions]
    
    axes[0, 1].plot(positions, avg_attentions, 'b-o', label='Avg Attention Received')
    axes[0, 1].plot(positions, avg_values, 'r-s', label='Avg Value Magnitude')
    axes[0, 1].set_xlabel('Token Position')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title('Attention vs Values by Position')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Layer-wise sink score analysis
    layer_data = defaultdict(list)
    for d in attention_data:
        layer_data[d['layer']].append(d['sink_score'])
    
    layers_sorted = sorted(layer_data.keys())
    sink_scores = [layer_data[l] for l in layers_sorted]
    
    axes[1, 0].boxplot(sink_scores, labels=layers_sorted)
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Sink Score (Attention/Value)')
    axes[1, 0].set_title('Sink Score Distribution by Layer')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Projection matrix analysis
    if projection_hook.projection_analysis:
        proj_data = projection_hook.projection_analysis
        layers = sorted(proj_data.keys())
        min_values = []
        mean_values = []
        zero_likes = []
        
        for layer in layers:
            data = proj_data[layer]
            min_values.append(data.get('min_value_norm', 0))
            mean_values.append(data.get('mean_value_norm', 0))
            zero_likes.append(data.get('zero_like_values', 0))
        
        axes[1, 1].plot(layers, min_values, 'r-o', label='Min Value Norm')
        axes[1, 1].plot(layers, mean_values, 'b-s', label='Mean Value Norm')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Value Magnitude')
        axes[1, 1].set_title('Projection Matrix Value Norms by Layer')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_sink_analysis.png'), dpi=300, bbox_inches='tight')
    logger.info(f"Value sink visualization saved to {output_dir}/value_sink_analysis.png")
    plt.show()


def analyze_sink_patterns(value_hook, projection_hook, output_file=None):
    """Analyze and report sink patterns from collected data"""
    logger.info("\n=== VALUE SINK PATTERN ANALYSIS ===")
    
    attention_data = value_hook.attention_vs_values
    if not attention_data:
        logger.warning("No attention data available for analysis")
        return
    
    # 1. Overall statistics
    total_positions = len(attention_data)
    potential_sinks = [d for d in attention_data if d['is_potential_sink']]
    sink_percentage = len(potential_sinks) / total_positions * 100
    
    logger.info(f"Total position-head combinations analyzed: {total_positions}")
    logger.info(f"Potential sinks identified: {len(potential_sinks)} ({sink_percentage:.2f}%)")
    
    # 2. Position-based analysis
    pos_sink_counts = defaultdict(int)
    pos_total_counts = defaultdict(int)
    
    for d in attention_data:
        pos = d['position']
        pos_total_counts[pos] += 1
        if d['is_potential_sink']:
            pos_sink_counts[pos] += 1
    
    logger.info("\nSink behavior by token position:")
    for pos in sorted(pos_total_counts.keys())[:10]:  # First 10 positions
        sink_rate = pos_sink_counts[pos] / pos_total_counts[pos] * 100
        logger.info(f"  Position {pos}: {sink_rate:.1f}% sink rate ({pos_sink_counts[pos]}/{pos_total_counts[pos]})")
    
    # 3. Layer-based analysis
    layer_sink_counts = defaultdict(int)
    layer_total_counts = defaultdict(int)
    
    for d in attention_data:
        layer = d['layer']
        layer_total_counts[layer] += 1
        if d['is_potential_sink']:
            layer_sink_counts[layer] += 1
    
    logger.info("\nSink behavior by layer:")
    for layer in sorted(layer_total_counts.keys()):
        sink_rate = layer_sink_counts[layer] / layer_total_counts[layer] * 100
        logger.info(f"  Layer {layer}: {sink_rate:.1f}% sink rate ({layer_sink_counts[layer]}/{layer_total_counts[layer]})")
    
    # 4. Top sink candidates
    top_sinks = sorted(potential_sinks, key=lambda x: x['sink_score'], reverse=True)[:20]
    logger.info("\nTop 20 sink candidates (highest attention/value ratios):")
    for i, sink in enumerate(top_sinks[:10]):
        logger.info(f"  {i+1}. L{sink['layer']}H{sink['head']} Pos{sink['position']} '{sink['token']}': "
                   f"Score={sink['sink_score']:.2f} (A={sink['attention_received']:.3f}, V={sink['value_norm']:.3f})")
    
    # 5. Projection matrix insights
    if projection_hook.projection_analysis:
        logger.info("\nProjection Matrix Analysis:")
        for layer, data in projection_hook.projection_analysis.items():
            v_type = data.get('v_type', 'unknown')
            min_norm = data.get('min_value_norm', 0)
            zero_likes = data.get('zero_like_values', 0)
            sample_size = projection_hook.sample_tokens
            
            logger.info(f"  Layer {layer} ({v_type}): Min norm={min_norm:.4f}, "
                       f"Zero-like values={zero_likes}/{sample_size} ({zero_likes/sample_size*100:.1f}%)")
    
    # Save detailed analysis
    if output_file:
        analysis_data = {
            'summary': {
                'total_positions': total_positions,
                'potential_sinks': len(potential_sinks),
                'sink_percentage': sink_percentage
            },
            'position_analysis': {pos: {'sink_rate': pos_sink_counts[pos] / pos_total_counts[pos] * 100,
                                       'sink_count': pos_sink_counts[pos],
                                       'total_count': pos_total_counts[pos]}
                                 for pos in pos_total_counts.keys()},
            'layer_analysis': {layer: {'sink_rate': layer_sink_counts[layer] / layer_total_counts[layer] * 100,
                                      'sink_count': layer_sink_counts[layer],
                                      'total_count': layer_total_counts[layer]}
                              for layer in layer_total_counts.keys()},
            'top_sinks': top_sinks,
            'projection_analysis': projection_hook.projection_analysis
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        logger.info(f"\nDetailed analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze value/projection matrices for attention sink behavior')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='value_sink_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--model-type', type=str, default='vanilla',
                        choices=['vanilla', 'symbolic', 'tft'])
    parser.add_argument('--prompt', type=str, default="The cat sat on the mat. The dog ran in the park. The bird flew in the sky.",
                        help='Text prompt for generation analysis')
    parser.add_argument('--max-tokens', type=int, default=10,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--value-threshold', type=float, default=0.1,
                        help='Threshold for considering values as "low magnitude"')
    parser.add_argument('--analyze-layers', type=int, nargs='+', default=None,
                        help='Specific layers to analyze (default: first 6)')
    parser.add_argument('--sample-tokens', type=int, default=100,
                        help='Number of tokens to sample for projection analysis')
    parser.add_argument('--save-analysis', type=str, default=None,
                        help='Path to save detailed analysis JSON')
    
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
    
    # Create specialized hooks
    analyze_layers = args.analyze_layers or list(range(min(6, config.n_layer)))
    
    value_hook = ValueSinkAnalysisHook(
        analyze_layers=analyze_layers,
        value_threshold=args.value_threshold
    )
    
    projection_hook = ProjectionMatrixAnalysisHook(
        model=model,
        tokenizer=tokenizer,
        sample_tokens=args.sample_tokens
    )
    
    hooks = [value_hook, projection_hook]
    
    logger.info(f"Analyzing value sink behavior with {len(hooks)} specialized hooks")
    logger.info(f"Target layers: {analyze_layers}")
    logger.info(f"Value threshold: {args.value_threshold}")
    
    # Run generation with analysis
    logger.info(f"\nRunning analysis on prompt: '{args.prompt}'")
    
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
    
    logger.info(f"\nGenerated text: {generated_text}")
    
    # Analyze results
    save_path = os.path.join(output_dir, 'sink_analysis.json') if args.save_analysis else None
    analyze_sink_patterns(value_hook, projection_hook, save_path)
    
    # Create visualizations
    try:
        create_value_sink_visualizations(value_hook, projection_hook, output_dir)
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        logger.info("Analysis completed without visualizations")
    
    logger.info(f"\nValue sink analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()