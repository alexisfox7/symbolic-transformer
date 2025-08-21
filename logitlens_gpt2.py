#!/usr/bin/env python3
"""
LogitLens implementation for GPT2LMHeadModel from transformers.

Usage:
    python logitlens_gpt2.py --text "The cat sat on the"
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Any

class GPT2LogitLens:
    """LogitLens implementation for GPT2LMHeadModel."""
    
    def __init__(self, model_name: str = 'gpt2', device: str = None):
        """Initialize with pretrained GPT2 model."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded: {self.model.config.n_layer} layers, {self.model.config.n_embd} dim")
    
    def run_logit_lens(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Run logit lens analysis on input text."""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(self.device)
        # Note: Not using attention_mask to match testss.py behavior
        
        # Decode input tokens for display
        input_tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0]]
        
        predictions = []
        
        with torch.no_grad():
            # Get embeddings
            inputs_embeds = self.model.transformer.wte(input_ids)
            position_embeds = self.model.transformer.wpe(torch.arange(input_ids.shape[1], device=self.device))
            hidden_states = inputs_embeds + position_embeds
            
            # Process through each layer
            for layer_idx in range(self.model.config.n_layer):
                # Apply transformer block (without explicit attention mask to match testss.py)
                block = self.model.transformer.h[layer_idx]
                hidden_states = block(hidden_states)[0]
                
                # Apply layer norm (like GPT2 does before final output)
                normalized_states = self.model.transformer.ln_f(hidden_states)
                
                # Apply language modeling head (unembedding)
                logits = self.model.lm_head(normalized_states)
                
                # Focus on last position (next token prediction)
                last_logits = logits[0, -1, :]  # [vocab_size]
                probs = F.softmax(last_logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, top_k)
                
                # Decode top tokens
                top_tokens = []
                for idx in top_indices:
                    token_text = self.tokenizer.decode([idx.item()])
                    # Handle special characters
                    if '\n' in token_text or len(token_text.strip()) == 0:
                        token_text = repr(token_text)
                    top_tokens.append(token_text)
                
                # Calculate entropy/perplexity
                entropy = -(probs * torch.log(probs + 1e-10)).sum()
                perplexity = torch.exp(entropy)
                
                # Store layer prediction
                layer_pred = {
                    'layer': layer_idx,
                    'top_tokens': top_tokens,
                    'top_probs': top_probs.cpu().tolist(),
                    'top_indices': top_indices.cpu().tolist(),
                    'entropy': entropy.item(),
                    'perplexity': perplexity.item(),
                    'most_likely': top_tokens[0],
                    'confidence': top_probs[0].item()
                }
                
                predictions.append(layer_pred)
            
            # Get final model prediction for comparison
            final_outputs = self.model(input_ids)
            final_logits = final_outputs.logits[0, -1, :]
            final_probs = F.softmax(final_logits, dim=-1)
            final_top_probs, final_top_indices = torch.topk(final_probs, top_k)
            
            final_top_tokens = []
            for idx in final_top_indices:
                token_text = self.tokenizer.decode([idx.item()])
                if '\n' in token_text or len(token_text.strip()) == 0:
                    token_text = repr(token_text)
                final_top_tokens.append(token_text)
            
            final_entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum()
            final_perplexity = torch.exp(final_entropy)
            
            final_prediction = {
                'layer': 'FINAL',
                'top_tokens': final_top_tokens,
                'top_probs': final_top_probs.cpu().tolist(),
                'top_indices': final_top_indices.cpu().tolist(),
                'entropy': final_entropy.item(),
                'perplexity': final_perplexity.item(),
                'most_likely': final_top_tokens[0],
                'confidence': final_top_probs[0].item()
            }
        
        return {
            'input_text': text,
            'input_tokens': input_tokens,
            'layer_predictions': predictions,
            'final_prediction': final_prediction,
            'num_layers': self.model.config.n_layer
        }
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot logit lens results."""
        predictions = results['layer_predictions']
        final_pred = results['final_prediction']
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Perplexity across layers
        layers = list(range(len(predictions))) + ['FINAL']
        perplexities = [p['perplexity'] for p in predictions] + [final_pred['perplexity']]
        
        ax1 = axes[0]
        ax1.plot(range(len(predictions)), perplexities[:-1], 'bo-', linewidth=2, markersize=6, label='Layer Output')
        ax1.plot(len(predictions), perplexities[-1], 'ro', markersize=8, label='Final Output')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity Across Layers')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(list(range(len(predictions))) + [len(predictions)])
        ax1.set_xticklabels([str(i) for i in range(len(predictions))] + ['FINAL'])
        
        # Plot 2: Top predictions
        ax2 = axes[1]
        all_preds = predictions + [final_pred]
        
        for i, pred in enumerate(all_preds):
            token = pred['most_likely']
            conf = pred['confidence']
            
            color = 'red' if pred['layer'] == 'FINAL' else 'lightblue'
            weight = 'bold' if pred['layer'] == 'FINAL' else 'normal'
            
            ax2.text(i, 0.5, f"'{token}'\n({conf:.3f})", 
                    ha='center', va='center', fontsize=9, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Top Prediction')
        ax2.set_title(f"Most Likely Next Token Across Layers\nInput: '{results['input_text']}'")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(all_preds)))
        ax2.set_xticklabels([str(i) for i in range(len(predictions))] + ['FINAL'])
        
        # Plot 3: Confidence (probability of top prediction)
        ax3 = axes[2]
        confidences = [p['confidence'] for p in predictions] + [final_pred['confidence']]
        
        ax3.plot(range(len(predictions)), confidences[:-1], 'go-', linewidth=2, markersize=6, label='Layer Confidence')
        ax3.plot(len(predictions), confidences[-1], 'ro', markersize=8, label='Final Confidence')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Confidence (Top Token Probability)')
        ax3.set_title('Prediction Confidence Across Layers')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xticks(list(range(len(predictions))) + [len(predictions)])
        ax3.set_xticklabels([str(i) for i in range(len(predictions))] + ['FINAL'])
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def print_results(self, results: Dict[str, Any]):
        """Print logit lens results in a readable format."""
        print(f"\nLogitLens Analysis for GPT2")
        print("=" * 80)
        print(f"Input: '{results['input_text']}'")
        print(f"Input tokens: {results['input_tokens']}")
        print("=" * 80)
        
        print(f"{'Layer':<6} {'Top Token':<15} {'Confidence':<12} {'Perplexity':<12}")
        print("-" * 50)
        
        # Print layer predictions
        for pred in results['layer_predictions']:
            layer = pred['layer']
            token = pred['most_likely']
            conf = pred['confidence']
            perp = pred['perplexity']
            print(f"{layer:<6} {token:<15} {conf:<12.4f} {perp:<12.2f}")
        
        # Print final prediction
        final = results['final_prediction']
        print("-" * 50)
        print(f"{'FINAL':<6} {final['most_likely']:<15} {final['confidence']:<12.4f} {final['perplexity']:<12.2f}")
        print("=" * 80)
        
        # Print top alternatives for final prediction
        print(f"\nFinal Top-{len(final['top_tokens'])} Predictions:")
        for i, (token, prob) in enumerate(zip(final['top_tokens'], final['top_probs'])):
            print(f"  {i+1}. '{token}' ({prob:.1%})")
        
        print(f"\n🎯 Prediction: '{results['input_text']}' → '{final['most_likely']}'")
        print(f"📊 Confidence: {final['confidence']:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Run LogitLens on GPT2')
    parser.add_argument('--text', type=str, default="Ben saw a dog. Ben saw a dog. Ben saw a",
                       help='Text to analyze')
    parser.add_argument('--model', type=str, default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='GPT2 model size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save plot (optional)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    # Initialize LogitLens
    logit_lens = GPT2LogitLens(model_name=args.model, device=args.device)
    
    # Run analysis
    print(f"\nAnalyzing: '{args.text}'")
    results = logit_lens.run_logit_lens(args.text, top_k=args.top_k)
    
    # Print results
    logit_lens.print_results(results)
    
    # Plot results
    if not args.no_plot:
        logit_lens.plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()