#!/usr/bin/env python3
"""
Analyze token representations X^ℓ at different layers ℓ.
Compare cosine similarities and track how representations evolve through the network.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from collections import defaultdict
import argparse
from pathlib import Path

# Run with: python -m rank_collapse_analyzer [args] or set PYTHONPATH before running

class TokenRepresentationAnalyzer:
    """Analyze how token representations change through transformer layers."""
    
    def __init__(self, checkpoint_path, model_type="vanilla"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.layer_representations = {}
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer."""
        from src.model import get_model
        from src.config import TransformerConfig
        from src.mytokenizers import create_tokenizer
        
        print(f"Loading model from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract config
        if 'config' in checkpoint:
            config_data = checkpoint['config']
            if hasattr(config_data, '__dict__'):
                config = config_data
            else:
                config = TransformerConfig(**config_data)
        else:
            raise ValueError("No config found in checkpoint")
        
        # Create model and tokenizer
        self.model = get_model(self.model_type, config)
        self.tokenizer = create_tokenizer('gpt2')
        config.update_from_tokenizer(self.tokenizer)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Model loaded: {self.model.get_num_params()/1e6:.2f}M parameters")
        print(f"Number of layers: {len(self.model.transformer.h) if hasattr(self.model, 'transformer') else 'Unknown'}")
        
        return self.model, self.tokenizer

    def extract_layer_representations(self, input_text_or_tokens, max_layers=None):
        """
        Extract token representations X^ℓ from each layer ℓ.
        
        Args:
            input_text_or_tokens: Either text string or token IDs
            max_layers: Maximum number of layers to analyze (None = all)
        """
        print(f"Extracting representations for: '{input_text_or_tokens}'")
        
        # Prepare input
        if isinstance(input_text_or_tokens, str):
            inputs = self.tokenizer.encode(input_text_or_tokens, return_tensors='pt')
            text = input_text_or_tokens
        else:
            inputs = torch.tensor([input_text_or_tokens]) if isinstance(input_text_or_tokens, list) else input_text_or_tokens
            text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
        
        print(f"Input tokens: {inputs[0].tolist()}")
        print(f"Decoded text: '{text}'")
        
        # Hook to capture layer outputs
        layer_outputs = {}
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    # For models that return (hidden_states, ...) tuples
                    if len(output) >= 2 and hasattr(output[0], 'shape'):
                        layer_outputs[layer_name] = output[1].detach().clone()
                    else:
                        layer_outputs[layer_name] = output[1].detach().clone()
                else:
                    layer_outputs[layer_name] = output.detach().clone()
            return hook_fn
        
        # Register hooks
        hooks = []
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            if max_layers:
                layers = layers[:max_layers]
            
            for i, layer in enumerate(layers):
                hook = layer.register_forward_hook(create_hook(f'layer_{i}'))
                hooks.append(hook)
        
        # Also capture initial embeddings and final output
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                # Capture token embeddings
                embed_hook = self.model.transformer.wte.register_forward_hook(create_hook('embeddings'))
                hooks.append(embed_hook)
            
            if hasattr(self.model.transformer, 'ln_f'):
                # Capture final layer norm output
                final_hook = self.model.transformer.ln_f.register_forward_hook(create_hook('final_norm'))
                hooks.append(final_hook)
        
        try:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
                
                # Also manually capture the final logits
                layer_outputs['logits'] = outputs['logits'].detach().clone()
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Store results
        self.layer_representations[text] = {
            'input_tokens': inputs[0].tolist(),
            'representations': layer_outputs,
            'sequence_length': inputs.shape[1]
        }
        
        print(f"Captured representations from {len(layer_outputs)} layers/components")
        return layer_outputs

    def compute_cosine_similarities(self, representations, position_idx=-1):
        """
        Compute cosine similarities between token representations.
        
        Args:
            representations: Dict of layer_name -> tensor representations
            position_idx: Which position to analyze (-1 = last token)
        """
        similarities = {}
        
        for layer_name, repr_tensor in representations.items():
            if repr_tensor is None or len(repr_tensor.shape) < 3:
                continue
                
            # repr_tensor shape: [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = repr_tensor.shape
            
            if position_idx == -1:
                position_idx = seq_len - 1
            elif position_idx >= seq_len:
                continue
            
            # Get representations at specific position: [batch_size, hidden_dim]
            position_repr = repr_tensor[:, position_idx, :]  # [1, hidden_dim]
            
            # For now, we'll compute self-similarity (preparation for multi-input comparison)
            # Normalize for cosine similarity
            normalized_repr = F.normalize(position_repr, dim=1)
            
            # Compute similarity with itself (will be 1.0, but useful for multi-input analysis)
            similarity_matrix = torch.mm(normalized_repr, normalized_repr.t())
            
            similarities[layer_name] = {
                'similarity_matrix': similarity_matrix,
                'representation': position_repr,
                'normalized_representation': normalized_repr,
                'norm': position_repr.norm(dim=1).item(),
                'position': position_idx
            }
        
        return similarities

    def compare_multiple_inputs(self, inputs_list, position_idx=-1):
        """
        Compare token representations across multiple inputs.
        
        Args:
            inputs_list: List of text strings or token sequences
            position_idx: Which position to analyze (-1 = last token)
        """
        print(f"\n🔍 Comparing representations across {len(inputs_list)} inputs:")
        for i, inp in enumerate(inputs_list):
            print(f"  {i+1}. '{inp}'")
        
        # Extract representations for each input
        all_representations = {}
        for i, inp in enumerate(inputs_list):
            repr_dict = self.extract_layer_representations(inp)
            all_representations[f'input_{i}'] = repr_dict
        
        # Compare representations layer by layer
        layer_comparison = {}
        
        # Get all layer names (use first input as reference)
        first_input_key = list(all_representations.keys())[0]
        layer_names = list(all_representations[first_input_key].keys())
        
        for layer_name in layer_names:
            print(f"\n📊 Analyzing layer: {layer_name}")
            
            # Collect representations from all inputs for this layer
            layer_reprs = []
            layer_norms = []
            
            for input_key in all_representations.keys():
                if layer_name in all_representations[input_key]:
                    repr_tensor = all_representations[input_key][layer_name]
                    
                    if repr_tensor is not None and len(repr_tensor.shape) >= 3:
                        batch_size, seq_len, hidden_dim = repr_tensor.shape
                        
                        # Get position
                        if position_idx == -1:
                            pos_idx = seq_len - 1
                        else:
                            pos_idx = min(position_idx, seq_len - 1)
                        
                        # Extract representation at position
                        pos_repr = repr_tensor[0, pos_idx, :]  # [hidden_dim]
                        layer_reprs.append(pos_repr)
                        layer_norms.append(pos_repr.norm().item())
            
            if len(layer_reprs) < 2:
                continue
            
            # Stack representations: [num_inputs, hidden_dim]
            stacked_reprs = torch.stack(layer_reprs)
            
            # Normalize for cosine similarity
            normalized_reprs = F.normalize(stacked_reprs, dim=1)
            
            # Compute pairwise cosine similarities
            similarity_matrix = torch.mm(normalized_reprs, normalized_reprs.t())
            
            # Get off-diagonal similarities (between different inputs)
            mask = ~torch.eye(len(layer_reprs), dtype=torch.bool)
            off_diagonal_sims = similarity_matrix[mask]
            
            layer_comparison[layer_name] = {
                'similarity_matrix': similarity_matrix,
                'mean_pairwise_similarity': off_diagonal_sims.mean().item(),
                'max_pairwise_similarity': off_diagonal_sims.max().item(),
                'min_pairwise_similarity': off_diagonal_sims.min().item(),
                'representation_norms': layer_norms,
                'mean_norm': np.mean(layer_norms),
                'std_norm': np.std(layer_norms)
            }
            
            print(f"  Mean pairwise similarity: {layer_comparison[layer_name]['mean_pairwise_similarity']:.4f}")
            print(f"  Max pairwise similarity: {layer_comparison[layer_name]['max_pairwise_similarity']:.4f}")
            print(f"  Mean representation norm: {layer_comparison[layer_name]['mean_norm']:.4f}")
        
        return layer_comparison

    def plot_similarity_progression(self, layer_comparison, save_path=None):
        """Plot how cosine similarities change through layers."""
        print(f"\n📈 Creating similarity progression plot...")
        
        # Extract layer-wise statistics
        layer_names = []
        mean_similarities = []
        max_similarities = []
        mean_norms = []
        
        # Sort layers by number for proper ordering
        sorted_layers = sorted(layer_comparison.items(), 
                              key=lambda x: self._extract_layer_number(x[0]))
        
        for layer_name, stats in sorted_layers:
            layer_names.append(layer_name)
            mean_similarities.append(stats['mean_pairwise_similarity'])
            max_similarities.append(stats['max_pairwise_similarity'])
            mean_norms.append(stats['mean_norm'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean similarities
        axes[0, 0].plot(range(len(layer_names)), mean_similarities, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Pairwise Cosine Similarity by Layer')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Cosine Similarity')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        #axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
        #axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
        axes[0, 0].legend()
        
        # Plot 2: Max similarities
        axes[0, 1].plot(range(len(layer_names)), max_similarities, 'o-', color='red', linewidth=2, markersize=6)
        axes[0, 1].set_title('Max Pairwise Cosine Similarity by Layer')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Max Cosine Similarity')
        axes[0, 1].set_xticks(range(len(layer_names)))
        axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
        axes[0, 1].legend()
        
        # Plot 3: Representation norms
        axes[1, 0].plot(range(len(layer_names)), mean_norms, 'o-', color='green', linewidth=2, markersize=6)
        axes[1, 0].set_title('Mean Representation Norm by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Mean L2 Norm')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Similarity heatmap for final layer
        if layer_comparison:
            final_layer = sorted_layers[-1][0]
            final_similarity_matrix = layer_comparison[final_layer]['similarity_matrix'].numpy()
            
            im = axes[1, 1].imshow(final_similarity_matrix, cmap='Blues', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Similarity Matrix - {final_layer}')
            axes[1, 1].set_xlabel('Input Index')
            axes[1, 1].set_ylabel('Input Index')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Add text annotations
            for i in range(final_similarity_matrix.shape[0]):
                for j in range(final_similarity_matrix.shape[1]):
                    text = axes[1, 1].text(j, i, f'{final_similarity_matrix[i, j]:.3f}',
                                         ha="center", va="center", color="black", fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()

    def _extract_layer_number(self, layer_name):
        """Extract layer number for sorting."""
        if 'layer_' in layer_name:
            try:
                return int(layer_name.split('_')[1])
            except:
                pass
        
        # Special ordering for non-numbered layers
        order_map = {
            'embeddings': -1,
            'final_norm': 1000,
            'logits': 1001
        }
        return order_map.get(layer_name, 999)

    def analyze_rank_collapse_progression(self, inputs_list):
        """Analyze how rank collapse progresses through layers."""
        print(f"\n🔬 ANALYZING RANK COLLAPSE PROGRESSION")
        print("="*60)
        
        # Get layer comparison
        layer_comparison = self.compare_multiple_inputs(inputs_list)
        
        # Analyze progression
        print(f"\n📊 Layer-by-layer Analysis:")
        print("-" * 40)
        
        collapse_progression = []
        
        sorted_layers = sorted(layer_comparison.items(), 
                              key=lambda x: self._extract_layer_number(x[0]))
        
        for layer_name, stats in sorted_layers:
            mean_sim = stats['mean_pairwise_similarity']
            max_sim = stats['max_pairwise_similarity']
            
            # Assess collapse level for this layer
            if mean_sim > 0.8:
                level = "SEVERE"
            elif mean_sim > 0.6:
                level = "MODERATE"
            elif mean_sim > 0.4:
                level = "MILD"
            else:
                level = "HEALTHY"
            
            collapse_progression.append({
                'layer': layer_name,
                'mean_similarity': mean_sim,
                'max_similarity': max_sim,
                'level': level,
                'mean_norm': stats['mean_norm']
            })
            
            print(f"{layer_name:12} | {level:8} | Mean: {mean_sim:.4f} | Max: {max_sim:.4f} | Norm: {stats['mean_norm']:.4f}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        severe_layers = sum(1 for x in collapse_progression if x['level'] == 'SEVERE')
        moderate_layers = sum(1 for x in collapse_progression if x['level'] == 'MODERATE')
        total_layers = len(collapse_progression)
        
        print(f"  Severe collapse: {severe_layers}/{total_layers} layers")
        print(f"  Moderate collapse: {moderate_layers}/{total_layers} layers")
        
        if severe_layers > total_layers * 0.5:
            overall_health = "🚨 CRITICAL: Widespread severe collapse"
        elif severe_layers > 0 or moderate_layers > total_layers * 0.5:
            overall_health = "⚠️ CONCERNING: Significant collapse detected"
        else:
            overall_health = "✅ ACCEPTABLE: Limited collapse"
        
        print(f"  {overall_health}")
        
        return collapse_progression

def main():
    parser = argparse.ArgumentParser(description='Analyze token representations across layers')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--model-type', type=str, default='vanilla', 
                       choices=['vanilla', 'symbolic', 'tft'])
    parser.add_argument('--inputs', nargs='+', 
                       default=["The quick brown fox", "Python is great", "Hello world", "Machine learning", "Deep neural networks"],
                       help='Input texts to compare')
    parser.add_argument('--position', type=int, default=-1,
                       help='Token position to analyze (-1 = last token)')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--max-layers', type=int, help='Maximum layers to analyze')
    
    args = parser.parse_args()
    
    print(f"🔍 Token Representation Analysis")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Inputs to compare: {len(args.inputs)}")
    
    # Initialize analyzer
    analyzer = TokenRepresentationAnalyzer(args.checkpoint, args.model_type)
    
    try:
        # Load model
        analyzer.load_model()
        
        # Analyze rank collapse progression
        collapse_progression = analyzer.analyze_rank_collapse_progression(args.inputs)
        
        # Create visualization
        layer_comparison = analyzer.compare_multiple_inputs(args.inputs, args.position)
        analyzer.plot_similarity_progression(layer_comparison, args.save_plot)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()