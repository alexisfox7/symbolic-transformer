# src/inference/logit_lens.py
"""
Simple logit lens implementation that works with existing hook system.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from .hooks import InferenceHook
import matplotlib.pyplot as plt
import os


class LogitLensHook(InferenceHook):
    """Simple logit lens hook that analyzes predictions at each layer."""
    
    def __init__(self, model, tokenizer, top_k=3):
        super().__init__("logit_lens")
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.predictions = []  # Store predictions for each layer
        
        # Get the unembedding matrix (lm_head)
        self.lm_head = model.lm_head
        
        # Get final layer norm if it exists
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            self.layer_norm = model.transformer.ln_f
        else:
            self.layer_norm = None
    
    def clear(self):
        """Clear stored predictions."""
        self.predictions = []
        self.data = []
    
    def analyze_layer(self, hidden_state, layer_idx, position, tokens):
        """Analyze predictions at a specific layer."""
        with torch.no_grad():
            # Apply layer norm if available (like original logit lens)
            if self.layer_norm:
                h = self.layer_norm(hidden_state)
                print("Layer norm applied for logit lens analysis")
            else:
                h = hidden_state
            
            # Get logits by applying unembedding matrix
            logits = self.lm_head(h)  # Shape: [batch, seq, vocab]
            
            # Focus on last position for generation
            last_logits = logits[0, -1, :]  # Shape: [vocab]
            
            # Get top-k predictions
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, self.top_k)
            
            # Decode tokens
            top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
            
            # Store prediction
            prediction = {
                'layer': layer_idx,
                'position': position,
                'tokens': top_tokens,
                'probs': top_probs.tolist(),
                'perplexity': torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item()
            }
            
            self.predictions.append(prediction)
            return prediction


def run_logit_lens_analysis(model, tokenizer, text, device):
    """
    Simple function to run logit lens analysis on text.
    
    Args:
        model: Your transformer model
        tokenizer: Tokenizer
        text: Text to analyze
        device: Device to run on
        
    Returns:
        Tuple of (predictions per layer, final prediction info)
    """
    from .hooks import InferenceHookManager
    
    # Create logit lens hook
    logit_hook = LogitLensHook(model, tokenizer, top_k=5)
    
    # Set up hook manager
    hook_manager = InferenceHookManager()
    hook_manager.add_hook(logit_hook)
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
    
    return logit_hook.predictions, final_prediction


def plot_logit_lens(predictions, final_prediction, save_path=None):
    """Simple plotting function for logit lens results."""
    if not predictions:
        print("No predictions to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot perplexity (including final)
    layers = [p['layer'] for p in predictions] + ['FINAL']
    perplexities = [p['perplexity'] for p in predictions] + [final_prediction['perplexity']]
    
    # Convert layer names to numbers for plotting
    layer_nums = list(range(len(predictions))) + [len(predictions)]
    
    ax1.plot(layer_nums[:-1], perplexities[:-1], 'bo-', linewidth=2, markersize=6, label='Intermediate Layers')
    ax1.plot(layer_nums[-1], perplexities[-1], 'ro', markersize=8, label='Final Output')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity Across Layers')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(layer_nums)
    ax1.set_xticklabels([str(i) for i in range(len(predictions))] + ['FINAL'])
    
    # Plot top predictions (including final)
    all_predictions = predictions + [final_prediction]
    for i, pred in enumerate(all_predictions):
        top_token = pred['tokens'][0]
        top_prob = pred['probs'][0]
        
        if pred['layer'] == 'FINAL':
            # Highlight final prediction
            ax2.text(i, 0.5, f"'{top_token}'\n({top_prob:.3f})", 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
        else:
            ax2.text(i, 0.5, f"'{top_token}'\n({top_prob:.3f})", 
                    ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Top Prediction')
    ax2.set_title(f"Top Prediction Across Layers\nInput: '{final_prediction['input_text']}' → Final: '{final_prediction['predicted_next']}'")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(all_predictions)))
    ax2.set_xticklabels([str(i) for i in range(len(predictions))] + ['FINAL'])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Logit lens plot saved to: {save_path}")
    
    plt.show()


def print_logit_lens_analysis(predictions, final_prediction, text):
    """Print analysis results in a readable format."""
    print(f"\nLogit Lens Analysis for: '{text}'")
    print("=" * 70)
    print(f"{'Layer':<8} {'Top Token':<15} {'Probability':<12} {'Perplexity':<12}")
    print("-" * 70)
    
    # Print intermediate layers
    for pred in predictions:
        layer = pred['layer']
        top_token = pred['tokens'][0]
        top_prob = pred['probs'][0]
        perplexity = pred['perplexity']
        
        print(f"{layer:<8} {top_token:<15} {top_prob:<12.4f} {perplexity:<12.2f}")
    
    # Print final prediction prominently
    print("-" * 70)
    print(f"{'FINAL':<8} {final_prediction['tokens'][0]:<15} {final_prediction['probs'][0]:<12.4f} {final_prediction['perplexity']:<12.2f}")
    print("=" * 70)
    
    # Summary
    print(f"\n🎯 FINAL PREDICTION:")
    print(f"   Input:  '{text}'")
    print(f"   Predicts next token: '{final_prediction['predicted_next']}'")
    print(f"   Confidence: {final_prediction['probs'][0]:.1%}")
    
    # Show top 3 alternatives
    print(f"\n📊 Top 3 alternatives:")
    for i, (token, prob) in enumerate(zip(final_prediction['tokens'][:3], final_prediction['probs'][:3])):
        print(f"   {i+1}. '{token}' ({prob:.1%})")
    
    print("=" * 70)