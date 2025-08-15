#!/usr/bin/env python3
"""
Visualize attention patterns for GPT-2 default model.
Based on the existing attention visualization patterns in the codebase.
"""


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from collections import defaultdict


def extract_attention_from_gpt2(model, tokenizer, text, device='cpu'):
   """Extract attention patterns from GPT-2 model."""
   # Tokenize input
   inputs = tokenizer(text, return_tensors='pt').to(device)
   input_ids = inputs['input_ids']
  
   # Get tokens for visualization
   tokens = [tokenizer.decode([id]) for id in input_ids[0]]
  
   print(f"Input text: '{text}'")
   print(f"Tokens: {tokens}")
  
   # Forward pass with attention output
   with torch.no_grad():
       outputs = model(input_ids, output_attentions=True)
       attentions = outputs.attentions  # List of attention tensors for each layer
  
   # Convert to format similar to existing visualization code
   attention_data = []
  
   for layer_idx, layer_attention in enumerate(attentions):
       # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
       layer_attention = layer_attention[0]  # Remove batch dimension
      
       for head_idx in range(layer_attention.shape[0]):
           head_attention = layer_attention[head_idx]  # [seq_len, seq_len]
          
           attention_data.append({
               'layer': layer_idx,
               'head': head_idx,
               'attention_matrix': head_attention.cpu(),
               'tokens': tokens,
               'position': len(tokens) - 1  # Final position
           })
  
   return attention_data, tokens


def create_attention_visualization(attention_data, tokens, output_dir=None, max_layers=3, max_heads=4, exclude_first_n=0):
   """
   Create attention matrix visualizations similar to existing codebase style.
   Based on tests/run_inference_with_hooks.py:create_attention_matrices_visualization
   """
   print("Creating attention matrix visualizations...")
  
   if not attention_data:
       print("No attention data available for visualization")
       return
  
   # Group attention data by layer and head
   attention_by_layer_head = defaultdict(list)
   for record in attention_data:
       key = (record['layer'], record['head'])
       attention_by_layer_head[key].append(record)
  
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
      
       # Use the record for this layer/head
       record = records[0]
       attention_matrix = record['attention_matrix'].numpy()
       display_tokens = record.get('tokens', [])
      
       # Exclude first N tokens if specified
       if exclude_first_n > 0:
           seq_len = attention_matrix.shape[0]
           if seq_len > exclude_first_n:
               attention_matrix = attention_matrix[exclude_first_n:, exclude_first_n:]
               display_tokens = display_tokens[exclude_first_n:]
               print(f"Layer {layer}, Head {head}: Excluded first {exclude_first_n} tokens, showing {attention_matrix.shape[0]} x {attention_matrix.shape[1]} matrix")
           else:
               print(f"Layer {layer}, Head {head}: Sequence too short ({seq_len}) to exclude {exclude_first_n} tokens")
      
       # Create heatmap with power scaling like in original code
       power = 0.8
       plot_matrix = np.power(attention_matrix, power)
  
       im = ax.imshow(plot_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
       cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       cbar.set_label('Attention Weight', rotation=270, labelpad=15)
      
       ax.set_xticks(range(len(display_tokens)))
       ax.set_yticks(range(len(display_tokens)))
       ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=8)
       ax.set_yticklabels(display_tokens, fontsize=8)


       ax.set_xlabel('Key Position')
       ax.set_ylabel('Query Position')
      
       # Update title to indicate exclusion if applicable
       if exclude_first_n > 0:
           ax.set_title(f'Layer {layer}, Head {head}\nExcluding first {exclude_first_n} tokens')
       else:
           ax.set_title(f'Layer {layer}, Head {head}\nGPT-2 Attention')
      
       ax.grid(True, alpha=0.3)
  
   # Hide unused subplots
   for idx in range(len(layer_head_pairs), len(axes)):
       axes[idx].set_visible(False)
  
   title_suffix = f" (excluding first {exclude_first_n} tokens)" if exclude_first_n > 0 else ""
   plt.suptitle(f'GPT-2 Attention Weight Matrices{title_suffix}', fontsize=16, fontweight='bold', y=0.99)
   plt.tight_layout(rect=[0, 0, 1, 0.99])
  
   if output_dir:
       os.makedirs(output_dir, exist_ok=True)
       filename = f'gpt2_attention_matrices_exclude_{exclude_first_n}.png' if exclude_first_n > 0 else 'gpt2_attention_matrices.png'
       save_path = os.path.join(output_dir, filename)
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
       print(f"Attention matrices saved to {save_path}")
  
   plt.show()


def analyze_attention_patterns(attention_data, tokens):
   """Analyze attention patterns similar to existing codebase."""
   print("\n=== GPT-2 Attention Pattern Analysis ===")
  
   # Analyze attention by layer/head
   layer_head_stats = {}
   for record in attention_data:
       key = (record['layer'], record['head'])
       if key not in layer_head_stats:
           layer_head_stats[key] = {
               'avg_weight': 0,
               'max_weight': 0,
               'entropy': 0
           }
      
       stats = layer_head_stats[key]
       attention_matrix = record['attention_matrix']
      
       # Calculate statistics for last token's attention (most interesting for generation)
       last_token_attention = attention_matrix[-1, :]
       stats['avg_weight'] = last_token_attention.mean().item()
       stats['max_weight'] = last_token_attention.max().item()
      
       # Calculate entropy (measure of attention spread)
       attention_probs = F.softmax(last_token_attention, dim=0)
       stats['entropy'] = -(attention_probs * torch.log(attention_probs + 1e-10)).sum().item()
  
   print("\nAttention statistics by layer/head (for last token):")
   for (layer, head), stats in sorted(layer_head_stats.items())[:12]:  # Show first 12
       print(f"  Layer {layer}, Head {head}: "
              f"avg weight: {stats['avg_weight']:.4f}, "
              f"max weight: {stats['max_weight']:.4f}, "
              f"entropy: {stats['entropy']:.4f}")
  
   return layer_head_stats


def main():
   # Configuration
   text = "Ben saw a dog. Ben saw a dog. Ben saw a"
   output_dir = "gpt2_attention_visualization"
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
   print(f"Loading GPT-2 model on {device}...")
  
   # Load GPT-2 model and tokenizer
   model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   model.to(device)
   model.eval()
  
   print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
  
   # Extract attention patterns
   attention_data, tokens = extract_attention_from_gpt2(model, tokenizer, text, device)
  
   # Analyze patterns
   stats = analyze_attention_patterns(attention_data, tokens)
  
   # Create visualizations
   print("\n=== Creating Visualizations ===")
  
   # Full attention matrices - all 12 layers
   create_attention_visualization(
       attention_data,
       tokens,
       output_dir,
       max_layers=12,
       max_heads=12,
       exclude_first_n=0
   )
  
   # Excluding first few tokens to focus on interesting patterns - all layers
   create_attention_visualization(
       attention_data,
       tokens,
       output_dir,
       max_layers=12,
       max_heads=12,
       exclude_first_n=3
   )
  
   print(f"\nVisualization complete! Check {output_dir} for results.")


if __name__ == "__main__":
   main()

