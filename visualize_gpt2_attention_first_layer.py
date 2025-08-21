#!/usr/bin/env python3
"""
Visualize attention patterns for GPT-2 default model with first-layer transformed embeddings.
Based on the existing attention visualization patterns in the codebase.
The attention matrices will be 3x the height to show original, word decompositions, and dark matter.
"""


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from collections import defaultdict


def get_token_through_first_layer(model, token_id, position_idx, seq_length, device='cuda'):
    """
    Pass a single token through the first transformer layer at a specific position.
    
    Args:
        token_id: The token to transform
        position_idx: Position where the token should be placed
        seq_length: Total sequence length (for positional encoding)
    
    Returns:
        The output of the first layer for that token
    """
    # Create dummy input with just our token at the specified position
    # Use padding token (50256) for other positions
    pad_token_id = 50256
    dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=device)
    dummy_input[0, position_idx] = token_id
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(dummy_input)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        
        # Initial hidden state (what goes into layer 0)
        hidden_states = inputs_embeds + position_embeds
        
        # Pass through first transformer block only
        first_block = model.transformer.h[0]
        
        # Create attention mask (attend only to the token position)
        attention_mask = torch.zeros((1, seq_length), device=device)
        attention_mask[0, position_idx] = 1.0
        
        # Extended attention mask for the transformer block
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through first layer
        outputs = first_block(
            hidden_states,
            attention_mask=extended_attention_mask,
        )
        
        first_layer_output = outputs[0]
        
        return first_layer_output[0, position_idx, :]


def get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx):
    """
    Simpler version using HuggingFace's output_hidden_states.
    Now using first-layer transformed embeddings for better alignment.
    """
    device = input_ids.device
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # hidden_states includes embeddings as layer 0, so layer_idx+1
        if layer_idx == -1:
            # Initial embeddings
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            # After layer layer_idx (so index is layer_idx + 1)
            residual_at_pos = outputs.hidden_states[layer_idx + 1][0, position_idx, :]
    
    # Apply final LN and project (standard LogitLens)
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    # Get top 3
    top3_indices = torch.topk(logits[0], k=3).indices
    
    # Use first-layer transformed embeddings instead of raw embeddings
    top3_embeddings = []
    for token_id in top3_indices:
        transformed = get_token_through_first_layer(
            model, token_id.item(), position_idx, seq_length, device
        )
        top3_embeddings.append(transformed)
    top3_embeddings = torch.stack(top3_embeddings)
    
    decomposed = {
        'original': residual_at_pos,
        'word1': top3_embeddings[0],
        'word2': top3_embeddings[1],
        'word3': top3_embeddings[2],
        'dark_matter': residual_at_pos - top3_embeddings.sum(dim=0)
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    return decomposed, token_names


def compute_attention_with_modified_query(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    modified_query_vector,
    decomposition_type
):
    """
    Compute attention pattern when replacing query vector at specific position.
    """
    # Get the attention layer
    attn_layer = model.transformer.h[layer_idx].attn
    
    # First, get the key and value vectors normally
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # CRUCIAL: Apply layer norm before attention (this is what GPT-2 does!)
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute Q, K, V from the NORMED hidden states
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Now handle the modification
        if decomposition_type != 'original':
            # For decomposed vectors, we need to apply layer norm too!
            modified_vector_normed = model.transformer.h[layer_idx].ln_1(
                modified_query_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project modified vector to QKV space
            modified_qkv = attn_layer.c_attn(modified_vector_normed.unsqueeze(0).unsqueeze(0))
            modified_q = modified_qkv[:, :, :hidden_dim]
            
            modified_q = modified_q.view(1, 1, num_heads, head_dim).transpose(1, 2)
            
            # Replace Q at the query position
            q[:, :, query_position, :] = modified_q[0, :, 0, :]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Extract attention pattern for specific head and position
        attention_pattern = attention_weights[0, head_idx, query_position, :]
        
    return attention_pattern


def extract_attention_from_gpt2(model, tokenizer, text, device='cpu'):
   """Extract attention patterns from GPT-2 model with decomposed queries."""
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
  
   # Convert to format for decomposed attention visualization
   attention_data = []
  
   for layer_idx, layer_attention in enumerate(attentions):
       # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
       layer_attention = layer_attention[0]  # Remove batch dimension
      
       for head_idx in range(layer_attention.shape[0]):
           # Use residual decomposition for the last position
           position_idx = len(tokens) - 1
           decomposed, token_names = get_residual_and_decompose_simple(
               model, input_ids, layer_idx, position_idx
           )
           
           # Compute attention patterns for each decomposition
           attention_patterns = {}
           
           for rep_name, rep_vector in decomposed.items():
               attention_pattern = compute_attention_with_modified_query(
                   model, input_ids, layer_idx, head_idx, position_idx, rep_vector, rep_name
               )
               attention_patterns[rep_name] = attention_pattern
           
           attention_data.append({
               'layer': layer_idx,
               'head': head_idx,
               'attention_patterns': attention_patterns,  # Dict of different query types
               'tokens': tokens,
               'position': position_idx,
               'decomposed': decomposed,
               'vocab_tokens': token_names
           })
  
   return attention_data, tokens


def create_attention_visualization(attention_data, tokens, output_dir=None, max_layers=3, max_heads=4, exclude_first_n=0):
   """
   Create attention matrix visualizations with 3x height for decomposed patterns.
   Shows original, word1, word2, word3, and dark_matter attention patterns stacked vertically.
   """
   print("Creating attention matrix visualizations with decomposed queries...")
  
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
  
   # Make figure 3x taller to accommodate 5 attention patterns stacked vertically
   fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 12))

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
       attention_patterns = record['attention_patterns']
       display_tokens = record.get('tokens', [])
       vocab_tokens = record.get('vocab_tokens', ['?', '?', '?'])
      
       # Exclude first N tokens if specified
       if exclude_first_n > 0:
           seq_len = len(display_tokens)
           if seq_len > exclude_first_n:
               display_tokens = display_tokens[exclude_first_n:]
               # Also trim the attention patterns
               for key in attention_patterns:
                   attention_patterns[key] = attention_patterns[key][exclude_first_n:]
               print(f"Layer {layer}, Head {head}: Excluded first {exclude_first_n} tokens, showing {len(display_tokens)} positions")
           else:
               print(f"Layer {layer}, Head {head}: Sequence too short ({seq_len}) to exclude {exclude_first_n} tokens")
      
       # Stack attention patterns vertically: original, word1, word2, word3, dark_matter
       pattern_order = ['original', 'word1', 'word2', 'word3', 'dark_matter']
       pattern_labels = ['Original', f"Word1: '{vocab_tokens[0]}'", f"Word2: '{vocab_tokens[1]}'", 
                        f"Word3: '{vocab_tokens[2]}'", 'Dark Matter']
       
       # Create combined matrix (5 x seq_len)
       seq_len = len(display_tokens)
       combined_matrix = np.zeros((5, seq_len))
       
       for i, pattern_name in enumerate(pattern_order):
           if pattern_name in attention_patterns:
               pattern = attention_patterns[pattern_name].cpu().numpy()
               combined_matrix[i, :] = pattern
       
       # Create heatmap
       im = ax.imshow(combined_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
       cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       cbar.set_label('Attention Weight', rotation=270, labelpad=15)
      
       # Set ticks and labels
       ax.set_xticks(range(len(display_tokens)))
       ax.set_yticks(range(5))
       ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=8)
       ax.set_yticklabels(pattern_labels, fontsize=8)

       ax.set_xlabel('Key Position')
       ax.set_ylabel('Query Type')
      
       # Update title
       if exclude_first_n > 0:
           ax.set_title(f'Layer {layer}, Head {head}\nDecomposed Attention (Excluding first {exclude_first_n} tokens)')
       else:
           ax.set_title(f'Layer {layer}, Head {head}\nDecomposed Attention (First Layer Embeddings)')
      
       ax.grid(True, alpha=0.3)
  
   # Hide unused subplots
   for idx in range(len(layer_head_pairs), len(axes)):
       axes[idx].set_visible(False)
  
   title_suffix = f" (excluding first {exclude_first_n} tokens)" if exclude_first_n > 0 else ""
   plt.suptitle(f'GPT-2 Decomposed Attention Weight Matrices{title_suffix}', fontsize=16, fontweight='bold', y=0.99)
   plt.tight_layout(rect=[0, 0, 1, 0.99])
  
   if output_dir:
       os.makedirs(output_dir, exist_ok=True)
       filename = f'gpt2_attention_matrices_decomposed_exclude_{exclude_first_n}.png' if exclude_first_n > 0 else 'gpt2_attention_matrices_decomposed.png'
       save_path = os.path.join(output_dir, filename)
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
       print(f"Decomposed attention matrices saved to {save_path}")
  
   plt.show()


def analyze_attention_patterns(attention_data, tokens):
   """Analyze decomposed attention patterns."""
   print("\n=== GPT-2 Decomposed Attention Pattern Analysis ===")
  
   # Analyze attention by layer/head
   layer_head_stats = {}
   for record in attention_data:
       key = (record['layer'], record['head'])
       if key not in layer_head_stats:
           layer_head_stats[key] = {
               'original_entropy': 0,
               'word_entropies': [],
               'dark_matter_entropy': 0,
               'vocab_tokens': record.get('vocab_tokens', ['?', '?', '?'])
           }
      
       stats = layer_head_stats[key]
       attention_patterns = record['attention_patterns']
      
       # Calculate entropy for each pattern type
       for pattern_name, pattern in attention_patterns.items():
           attention_probs = F.softmax(pattern, dim=0)
           entropy = -(attention_probs * torch.log(attention_probs + 1e-10)).sum().item()
           
           if pattern_name == 'original':
               stats['original_entropy'] = entropy
           elif pattern_name.startswith('word'):
               stats['word_entropies'].append(entropy)
           elif pattern_name == 'dark_matter':
               stats['dark_matter_entropy'] = entropy
  
   print("\nDecomposed attention statistics by layer/head:")
   for (layer, head), stats in sorted(layer_head_stats.items())[:12]:  # Show first 12
       avg_word_entropy = np.mean(stats['word_entropies']) if stats['word_entropies'] else 0
       vocab_str = ', '.join([f"'{token}'" for token in stats['vocab_tokens']])
       print(f"  Layer {layer}, Head {head}: vocab=[{vocab_str}]")
       print(f"    Original entropy: {stats['original_entropy']:.3f}")
       print(f"    Avg word entropy: {avg_word_entropy:.3f}")
       print(f"    Dark matter entropy: {stats['dark_matter_entropy']:.3f}")
  
   return layer_head_stats


def main():
   # Configuration
   text = "Ben laughed funny. Ben laughed funny. Ben laughed"
   output_dir = "gpt2_attention_visualization"
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
   print(f"Loading GPT-2 model on {device}...")
  
   # Load GPT-2 model and tokenizer
   model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   model.to(device)
   model.eval()
  
   print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
  
   # Extract attention patterns with decomposition
   attention_data, tokens = extract_attention_from_gpt2(model, tokenizer, text, device)
  
   # Analyze patterns
   stats = analyze_attention_patterns(attention_data, tokens)
  
   # Create visualizations
   print("\n=== Creating Decomposed Visualizations ===")
  
   # Full attention matrices - first 3 layers, 4 heads each
   create_attention_visualization(
       attention_data,
       tokens,
       output_dir,
       max_layers=3,
       max_heads=4,
       exclude_first_n=0
   )
  
   # Excluding first few tokens to focus on interesting patterns
   create_attention_visualization(
       attention_data,
       tokens,
       output_dir,
       max_layers=3,
       max_heads=4,
       exclude_first_n=3
   )
  
   print(f"\nDecomposed visualization complete! Check {output_dir} for results.")


if __name__ == "__main__":
   main()