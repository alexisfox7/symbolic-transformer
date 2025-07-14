# src/inference/head_logit_lens.py
"""
Head-wise logit lens implementation that analyzes predictions from individual attention heads.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from .hooks import InferenceHook
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class HeadLogitLensHook(InferenceHook):
    """Head-wise logit lens hook that analyzes predictions from individual attention heads."""
    
    def __init__(self, model, tokenizer, top_k=3):
        super().__init__("head_logit_lens")
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.head_predictions = []  # Store predictions for each head at each layer
        
        # Get model architecture info
        self.n_heads = model.config.n_head
        self.n_embd = model.config.n_embd
        self.head_dim = self.n_embd // self.n_heads
        
        # Get the unembedding matrix (lm_head)
        self.lm_head = model.lm_head
        
        # Create head-specific unembedding matrices by chunking the full embedding
        # Each head gets 1/n_heads of the vocabulary embedding dimension
        self.vocab_embeddings = model.transformer.wte.weight  # [vocab_size, n_embd]
        
        # Split embedding matrix into head-specific chunks
        self.head_embeddings = []
        for h in range(self.n_heads):
            start_idx = h * self.head_dim
            end_idx = (h + 1) * self.head_dim
            head_emb = self.vocab_embeddings[:, start_idx:end_idx]  # [vocab_size, head_dim]
            self.head_embeddings.append(head_emb)
        
        # Get final layer norm if it exists
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            self.layer_norm = model.transformer.ln_f
        else:
            self.layer_norm = None
    
    def clear(self):
        """Clear stored predictions."""
        self.head_predictions = []
        self.data = []
    
    def analyze_layer(self, hidden_state, layer_idx, position, tokens):
        """Analyze predictions from each attention head at a specific layer."""
        with torch.no_grad():
            # Apply layer norm if available
            if self.layer_norm:
                h = self.layer_norm(hidden_state)
            else:
                h = hidden_state
            
            # Focus on last position for generation: [batch, seq, n_embd] -> [n_embd]
            last_hidden = h[0, -1, :]  # Shape: [n_embd]
            
            # Split hidden state into head-specific chunks
            head_states = last_hidden.view(self.n_heads, self.head_dim)  # [n_heads, head_dim]
            
            layer_head_predictions = []
            
            # Analyze each head separately
            for head_idx in range(self.n_heads):
                head_state = head_states[head_idx]  # [head_dim]
                head_embedding = self.head_embeddings[head_idx]  # [vocab_size, head_dim]
                
                # Compute logits using only this head's portion of the embedding
                head_logits = torch.matmul(head_state, head_embedding.T)  # [vocab_size]
                
                # Get top-k predictions for this head
                probs = F.softmax(head_logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, self.top_k)
                
                # Decode tokens
                top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                
                # Store prediction for this head
                head_prediction = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'position': position,
                    'tokens': top_tokens,
                    'probs': top_probs.tolist(),
                    'perplexity': torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item(),
                    'entropy': -(probs * torch.log(probs + 1e-10)).sum().item(),
                    'max_prob': top_probs[0].item()
                }
                
                layer_head_predictions.append(head_prediction)
            
            # Store all head predictions for this layer
            self.head_predictions.extend(layer_head_predictions)
            
            return layer_head_predictions


def run_head_logit_lens_analysis(model, tokenizer, text, device):
    """
    Run head-wise logit lens analysis on text.
    
    Args:
        model: Your transformer model
        tokenizer: Tokenizer
        text: Text to analyze
        device: Device to run on
        
    Returns:
        Tuple of (head predictions, final prediction info)
    """
    from .hooks import InferenceHookManager
    
    # Create head logit lens hook
    head_logit_hook = HeadLogitLensHook(model, tokenizer, top_k=5)
    
    # Set up hook manager
    hook_manager = InferenceHookManager()
    hook_manager.add_hook(head_logit_hook)
    model.set_hook_manager(hook_manager)
    
    # Tokenize input
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids.to(device)
    
    # Prepare hook state
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]
    hook_state = {'tokens': tokens, 'position': 0}
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, hook_state=hook_state)
    
    # Get final prediction from the model's actual output
    final_logits = outputs['logits'][0, -1, :]  # Last position logits
    final_probs = F.softmax(final_logits, dim=-1)
    final_top_probs, final_top_indices = torch.topk(final_probs, 5)
    final_top_tokens = [tokenizer.decode([idx.item()]) for idx in final_top_indices]
    
    final_prediction = {
        'layer': 'FINAL',
        'position': len(tokens) - 1,
        'tokens': final_top_tokens,
        'probs': final_top_probs.tolist(),
        'perplexity': torch.exp(-(final_probs * torch.log(final_probs + 1e-10)).sum()).item(),
        'input_text': text,
        'predicted_next': final_top_tokens[0]
    }
    
    # Clean up
    model.set_hook_manager(None)
    
    return head_logit_hook.head_predictions, final_prediction


def plot_head_heatmap(head_predictions, final_prediction, save_path=None):
    """Create a heatmap showing top predictions from each head at each layer."""
    if not head_predictions:
        print("No head predictions to plot")
        return
    
    # Organize data for heatmap
    layers = sorted(set(pred['layer'] for pred in head_predictions))
    n_heads = len(set(pred['head'] for pred in head_predictions if pred['layer'] == layers[0]))
    
    # Create matrices for different metrics
    confidence_matrix = np.zeros((len(layers), n_heads))
    perplexity_matrix = np.zeros((len(layers), n_heads))
    token_matrix = np.full((len(layers), n_heads), "", dtype=object)
    
    for pred in head_predictions:
        layer_idx = layers.index(pred['layer'])
        head_idx = pred['head']
        
        confidence_matrix[layer_idx, head_idx] = pred['max_prob']
        perplexity_matrix[layer_idx, head_idx] = pred['perplexity']
        token_matrix[layer_idx, head_idx] = pred['tokens'][0]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Confidence heatmap
    im1 = axes[0, 0].imshow(confidence_matrix, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Max Probability by Head and Layer')
    axes[0, 0].set_xlabel('Head')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_xticks(range(n_heads))
    axes[0, 0].set_yticks(range(len(layers)))
    axes[0, 0].set_yticklabels(layers)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Add text annotations for confidence
    for i in range(len(layers)):
        for j in range(n_heads):
            text = axes[0, 0].text(j, i, f'{confidence_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
    
    # Plot 2: Perplexity heatmap (lower is better)
    im2 = axes[0, 1].imshow(perplexity_matrix, cmap='Reds_r', aspect='auto')
    axes[0, 1].set_title('Perplexity by Head and Layer (Lower = Better)')
    axes[0, 1].set_xlabel('Head')
    axes[0, 1].set_ylabel('Layer')
    axes[0, 1].set_xticks(range(n_heads))
    axes[0, 1].set_yticks(range(len(layers)))
    axes[0, 1].set_yticklabels(layers)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Plot 3: Token predictions (text overlay)
    axes[1, 0].set_xlim(-0.5, n_heads - 0.5)
    axes[1, 0].set_ylim(-0.5, len(layers) - 0.5)
    axes[1, 0].set_title('Top Token Predictions by Head')
    axes[1, 0].set_xlabel('Head')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_xticks(range(n_heads))
    axes[1, 0].set_yticks(range(len(layers)))
    axes[1, 0].set_yticklabels(layers)
    
    # Add token text
    for i in range(len(layers)):
        for j in range(n_heads):
            token = token_matrix[i, j]
            confidence = confidence_matrix[i, j]
            # Color code by confidence
            color = plt.cm.Blues(confidence)
            axes[1, 0].text(j, i, f"'{token}'", ha="center", va="center", 
                           fontsize=8, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # Plot 4: Agreement with final prediction
    final_token = final_prediction['tokens'][0]
    agreement_matrix = np.zeros((len(layers), n_heads))
    
    for i in range(len(layers)):
        for j in range(n_heads):
            if token_matrix[i, j] == final_token:
                agreement_matrix[i, j] = 1
    
    im4 = axes[1, 1].imshow(agreement_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Agreement with Final Prediction: "{final_token}"')
    axes[1, 1].set_xlabel('Head')
    axes[1, 1].set_ylabel('Layer')
    axes[1, 1].set_xticks(range(n_heads))
    axes[1, 1].set_yticks(range(len(layers)))
    axes[1, 1].set_yticklabels(layers)
    
    # Add checkmarks for agreement
    for i in range(len(layers)):
        for j in range(n_heads):
            if agreement_matrix[i, j] == 1:
                axes[1, 1].text(j, i, '✓', ha="center", va="center", 
                               color="darkgreen", fontsize=16, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Head heatmap saved to: {save_path}")
    
    plt.show()


def print_head_analysis(head_predictions, final_prediction, text):
    """Print detailed head-wise analysis results."""
    print(f"\nHead-Wise Logit Lens Analysis for: '{text}'")
    print("=" * 80)
    print(f"Model Configuration: {len(set(pred['head'] for pred in head_predictions if pred['layer'] == 0))} heads")
    print(f"Final Prediction: '{final_prediction['predicted_next']}' ({final_prediction['probs'][0]:.1%})")
    print("=" * 80)
    
    # Group by layer
    layers = sorted(set(pred['layer'] for pred in head_predictions))
    
    for layer in layers:
        print(f"\n📊 LAYER {layer}")
        print("-" * 60)
        print(f"{'Head':<6} {'Top Token':<15} {'Prob':<8} {'Perplexity':<10} {'Match Final':<12}")
        print("-" * 60)
        
        layer_preds = [p for p in head_predictions if p['layer'] == layer]
        layer_preds.sort(key=lambda x: x['head'])
        
        for pred in layer_preds:
            head = pred['head']
            top_token = pred['tokens'][0]
            top_prob = pred['probs'][0]
            perplexity = pred['perplexity']
            matches_final = "✓" if top_token == final_prediction['tokens'][0] else "✗"
            
            print(f"{head:<6} {top_token:<15} {top_prob:<8.3f} {perplexity:<10.2f} {matches_final:<12}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    
    final_token = final_prediction['tokens'][0]
    total_predictions = len(head_predictions)
    matching_predictions = sum(1 for p in head_predictions if p['tokens'][0] == final_token)
    
    print(f"Total head predictions: {total_predictions}")
    print(f"Matching final prediction: {matching_predictions} ({matching_predictions/total_predictions:.1%})")
    
    # Per-layer agreement
    print(f"\nAgreement by layer:")
    for layer in layers:
        layer_preds = [p for p in head_predictions if p['layer'] == layer]
        layer_matches = sum(1 for p in layer_preds if p['tokens'][0] == final_token)
        print(f"  Layer {layer}: {layer_matches}/{len(layer_preds)} heads ({layer_matches/len(layer_preds):.1%})")
    
    # Most confident heads
    print(f"\nMost confident predictions:")
    confident_preds = sorted(head_predictions, key=lambda x: x['max_prob'], reverse=True)[:5]
    for i, pred in enumerate(confident_preds):
        print(f"  {i+1}. Layer {pred['layer']}, Head {pred['head']}: '{pred['tokens'][0]}' ({pred['max_prob']:.1%})")
    
    print("=" * 80)


def analyze_head_specialization(head_predictions):
    """Analyze which heads tend to predict similar things (head specialization)."""
    print(f"\n🔬 HEAD SPECIALIZATION ANALYSIS")
    print("-" * 50)
    
    # Group predictions by head across all layers
    head_tokens = {}
    for pred in head_predictions:
        head_id = pred['head']
        if head_id not in head_tokens:
            head_tokens[head_id] = []
        head_tokens[head_id].append(pred['tokens'][0])
    
    # Analyze consistency for each head
    print(f"Head consistency (same token across layers):")
    for head_id in sorted(head_tokens.keys()):
        tokens = head_tokens[head_id]
        unique_tokens = set(tokens)
        consistency = 1.0 - (len(unique_tokens) - 1) / len(tokens) if len(tokens) > 1 else 1.0
        most_common = max(unique_tokens, key=tokens.count) if unique_tokens else "N/A"
        
        print(f"  Head {head_id}: {consistency:.2f} consistency, prefers '{most_common}'")
    
    # Find heads that agree most often
    print(f"\nHead agreement patterns:")
    heads = sorted(head_tokens.keys())
    for i, head1 in enumerate(heads):
        for head2 in heads[i+1:]:
            tokens1 = head_tokens[head1]
            tokens2 = head_tokens[head2]
            
            # Calculate agreement rate
            agreements = sum(1 for t1, t2 in zip(tokens1, tokens2) if t1 == t2)
            agreement_rate = agreements / len(tokens1) if tokens1 else 0
            
            if agreement_rate > 0.5:  # Only show significant agreement
                print(f"  Head {head1} ↔ Head {head2}: {agreement_rate:.1%} agreement")