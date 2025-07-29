# src/inference/logit_lens.py
"""
Simple logit lens implementation that works with existing hook system.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from src.hooks.base import InferenceHook
import matplotlib.pyplot as plt
import os


class LogitLensHook(InferenceHook):
    """Logit lens hook decodes intermediate layer outputs for analysis."""
    
    def __init__(self, model, tokenizer, top_k=3):
        super().__init__("logit_lens")
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.predictions = [] # stores predictions for each layer
        self.final_predicted_token_id = None  # Store final layer prediction
        
        self.lm_head = model.lm_head # unembedding matrix
        self.layer_norm = model.transformer.ln_f

    def clear(self):
        self.predictions = []
        self.data = {}
        self.final_predicted_token_id = None
    
    def set_final_prediction(self, token_id):
        """Set the final layer's predicted token ID for similarity calculations."""
        self.final_predicted_token_id = token_id
    
    def on_layer_end(self, layer_idx: int, outputs: Any, state: Dict[str, Any]) -> None:
        """Analyze predictions at each layer using logit lens."""
        with torch.no_grad():
            # Get input_ids from state for context
            input_ids = state.get('input_ids')
            
            # outputs is the hidden state after this layer
            hidden_state = outputs  # Shape: [batch, seq, hidden_dim]
            
            # Apply layer norm if available (like original logit lens)
            if self.layer_norm:
                h = self.layer_norm(hidden_state)
            
            # Get logits by applying unembedding matrix
            logits = self.lm_head(h)  # Shape: [batch, seq, vocab]
            
            # Focus on last position for generation
            last_logits = logits[0, -1, :]  # Shape: [vocab]
            
            # Get top-k predictions
            probs = F.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, self.top_k)
            
            # Decode tokens
            top_tokens = []
            for idx in top_indices:
                try:
                    token_text = self.tokenizer.decode([idx.item()], skip_special_tokens=False)
                    token_text = repr(token_text) if '\n' in token_text or len(token_text.strip()) == 0 else token_text.strip()
                    top_tokens.append(token_text)
                except:
                    top_tokens.append(f"ID_{idx.item()}")
            
            # Calculate dot product similarities
            # Get the embeddings for current token and predicted token
            last_hidden = h[0, -1, :]  # Shape: [hidden_dim]
            
            # Current token embedding (last token in the sequence)
            current_token_id = input_ids[0, -1].item()
            current_token_emb = self.model.transformer.wte(torch.tensor([current_token_id], device=input_ids.device))
            current_token_emb = current_token_emb.squeeze(0)  # Shape: [hidden_dim]
            
            # Calculate similarities
            last_hidden_norm = F.normalize(last_hidden, dim=0)
            current_emb_norm = F.normalize(current_token_emb, dim=0)
            
            # Calculate similarity to current token
            sim_current = torch.dot(last_hidden_norm, current_emb_norm).item()
            
            # Calculate similarity to final layer's predicted token (if available)
            sim_predicted = None
            if self.final_predicted_token_id is not None:
                final_predicted_emb = self.model.transformer.wte(torch.tensor([self.final_predicted_token_id], device=input_ids.device))
                final_predicted_emb = final_predicted_emb.squeeze(0)  # Shape: [hidden_dim]
                final_predicted_norm = F.normalize(final_predicted_emb, dim=0)
                sim_predicted = torch.dot(last_hidden_norm, final_predicted_norm).item()
            
            # Baseline similarity with a random/different token for comparison
            # Use a common token like "the" (token_id = 262 in GPT-2) as baseline
            baseline_token_id = 262  # "the" in GPT-2 tokenizer
            # Make sure it's different from current and predicted tokens
            if baseline_token_id == current_token_id:
                baseline_token_id = 290  # "and" 
            if self.final_predicted_token_id is not None and baseline_token_id == self.final_predicted_token_id:
                baseline_token_id = 318  # "a"
                
            baseline_emb = self.model.transformer.wte(torch.tensor([baseline_token_id], device=input_ids.device))
            baseline_emb = baseline_emb.squeeze(0)  # Shape: [hidden_dim]
            baseline_norm = F.normalize(baseline_emb, dim=0)
            sim_baseline = torch.dot(last_hidden_norm, baseline_norm).item()
            
            # Store prediction
            prediction = {
                'layer': layer_idx,
                'position': hidden_state.shape[1] - 1,  # Last position
                'tokens': top_tokens,
                'probs': top_probs.tolist(),
                'token_ids': top_indices.tolist(),
                'perplexity': torch.exp(-(probs * torch.log(probs + 1e-10)).sum()).item(),
                'current_token_id': current_token_id,
                'current_token': self.tokenizer.decode([current_token_id], skip_special_tokens=False),
                'similarity_to_current': sim_current,
                'similarity_to_predicted': sim_predicted,
                'similarity_to_baseline': sim_baseline,
                'baseline_token_id': baseline_token_id,
                'baseline_token': self.tokenizer.decode([baseline_token_id], skip_special_tokens=False)
            }
            
            self.predictions.append(prediction)


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
    from src.hooks.base import HookManager
    
    # Tokenize input first
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_ids = input_ids.to(device)
    
    # First pass: get final prediction to set for similarity calculations
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get final prediction from the model's actual output
    if isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    
    final_logits = logits[0, -1, :]  # Last position logits
    final_probs = F.softmax(final_logits, dim=-1)
    final_top_probs, final_top_indices = torch.topk(final_probs, 5)
    
    # Create logit lens hook and set the final prediction
    logit_hook = LogitLensHook(model, tokenizer, top_k=5)
    logit_hook.set_final_prediction(final_top_indices[0].item())  # Set the top prediction
    
    # Set up hook manager
    hook_manager = HookManager()
    hook_manager.add_hook(logit_hook)
    model.set_hook_manager(hook_manager)
    
    # Second pass: run with hooks to get layer-by-layer analysis
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Decode tokens with error handling
    final_top_tokens = []
    for idx in final_top_indices:
        try:
            token_text = tokenizer.decode([idx.item()], skip_special_tokens=False)
            token_text = repr(token_text) if '\n' in token_text or len(token_text.strip()) == 0 else token_text.strip()
            final_top_tokens.append(token_text)
        except:
            final_top_tokens.append(f"ID_{idx.item()}")
    
    final_prediction = {
        'layer': 'FINAL',
        'position': input_ids.shape[1] - 1,  # Last position
        'tokens': final_top_tokens,
        'probs': final_top_probs.tolist(),
        'perplexity': torch.exp(-(final_probs * torch.log(final_probs + 1e-10)).sum()).item(),
        'input_text': text,
        'predicted_next': final_top_tokens[0]
    }
    
    # Clean up hook manager
    model.set_hook_manager(None)
    
    return logit_hook.predictions, final_prediction


def plot_logit_lens(predictions, final_prediction, save_path=None):
    """Simple plotting function for logit lens results."""
    if not predictions:
        print("No predictions to plot")
        return
    
    # Check if similarity data is available
    has_similarity = predictions[0].get('similarity_to_current') is not None
    n_plots = 3 if has_similarity else 2
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 10 + (4 if has_similarity else 0)))
    
    # Handle single vs multiple subplots
    if n_plots == 1:
        axes = [axes]
    elif n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
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
    
    # Plot similarities if available
    if has_similarity:
        sim_current = [p['similarity_to_current'] for p in predictions]
        sim_predicted = [p.get('similarity_to_predicted', 0.0) for p in predictions]
        sim_baseline = [p.get('similarity_to_baseline', 0.0) for p in predictions]
        
        ax3.plot(range(len(predictions)), sim_current, 'go-', linewidth=2, markersize=6, label='Similarity to Current Token')
        ax3.plot(range(len(predictions)), sim_predicted, 'mo-', linewidth=2, markersize=6, label='Similarity to Final Predicted Token')
        ax3.plot(range(len(predictions)), sim_baseline, 'ro--', linewidth=1.5, markersize=4, alpha=0.7, label='Similarity to Baseline Token')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_title('Token Embedding Similarities Across Layers')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xticks(range(len(predictions)))
        ax3.set_xticklabels([str(i) for i in range(len(predictions))])
        ax3.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Logit lens plot saved to: {save_path}")
    
    plt.show()


def print_logit_lens_analysis(predictions, final_prediction, text):
    """Print analysis results in a readable format."""
    print(f"\nLogit Lens Analysis for: '{text}'")
    print("=" * 120)
    print(f"{'Layer':<8} {'Top Token':<15} {'Probability':<12} {'Perplexity':<12} {'Sim(Current)':<12} {'Sim(Predicted)':<13} {'Sim(Baseline)':<13}")
    print("-" * 120)
    
    # Print intermediate layers
    for pred in predictions:
        layer = pred['layer']
        top_token = pred['tokens'][0]
        top_prob = pred['probs'][0]
        perplexity = pred['perplexity']
        sim_current = pred.get('similarity_to_current', 0.0)
        sim_predicted = pred.get('similarity_to_predicted', 0.0)
        sim_baseline = pred.get('similarity_to_baseline', 0.0)
        
        print(f"{layer:<8} {top_token:<15} {top_prob:<12.4f} {perplexity:<12.2f} {sim_current:<12.4f} {sim_predicted:<13.4f} {sim_baseline:<13.4f}")
    
    # Print final prediction prominently
    print("-" * 120)
    print(f"{'FINAL':<8} {final_prediction['tokens'][0]:<15} {final_prediction['probs'][0]:<12.4f} {final_prediction['perplexity']:<12.2f}")
    print("=" * 120)
    
    # Print current token info if available
    if predictions and 'current_token' in predictions[0]:
        baseline_token = predictions[0].get('baseline_token', 'unknown')
        print(f"\n📍 Current token: '{predictions[0]['current_token']}'")
        print(f"📊 Final predicted token: '{final_prediction['tokens'][0]}'")
        print(f"📏 Baseline token: '{baseline_token}'")
        print(f"   (Sim(Current) = similarity to current token embedding)")
        print(f"   (Sim(Predicted) = similarity to final layer's predicted token embedding)")
        print(f"   (Sim(Baseline) = similarity to baseline token '{baseline_token}' for comparison)")
    
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