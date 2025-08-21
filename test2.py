import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from typing import Dict, List, Tuple

def get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx):
    """
    Simpler version using HuggingFace's output_hidden_states.
    """
    device = input_ids.device
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # hidden_states includes embeddings as layer 0, so layer_idx+1
        if layer_idx == -1:
            # Initial embeddings
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            # Before layer layer_idx (so index is layer_idx, which is after layer_idx-1)
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
    
    # Apply layer normalization for LogitLens (all layers need ln_f for vocab projection)
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    # Get top 3
    top3_indices = torch.topk(logits[0], k=3).indices
    top3_embeddings = model.transformer.wte(top3_indices)
    
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

def compute_attention_with_modified_keys(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    key_position,
    modified_key_vector,
    decomposition_type
):
    """
    Compute attention pattern when replacing KEY vector at specific position.
    Shows which positions attend to this modified key.
    """
    # Get the attention layer
    attn_layer = model.transformer.h[layer_idx].attn
    
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Apply layer norm before attention
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute Q, K, V from normed hidden states
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Replace KEY at specific position
        if decomposition_type != 'original':
            # Apply layer norm to modified vector
            modified_vector_normed = model.transformer.h[layer_idx].ln_1(
                modified_key_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project to get KEY
            modified_qkv = attn_layer.c_attn(modified_vector_normed.unsqueeze(0).unsqueeze(0))
            modified_k = modified_qkv[:, :, hidden_dim:2*hidden_dim]
            
            modified_k = modified_k.view(1, 1, num_heads, head_dim).transpose(1, 2)
            
            # Replace K at the key position
            k[:, :, key_position, :] = modified_k[0, :, 0, :]
        
        # Compute attention scores - shows how much each position attends to our key
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Extract how much each position attends to the key_position
        attention_to_key = attention_weights[0, head_idx, :, key_position]
        
    return attention_to_key

def analyze_decomposed_attention(model, tokenizer, text, layer_idx=5, head_idx=0):
    """
    Main analysis function - now analyzing which positions attend to implanted keys.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer_idx}, Head {head_idx}")
    
    # Pick position to analyze (e.g., last token)
    key_position = len(tokens) - 1
    
    # Get decomposed representations
    decomposed, token_names = get_residual_and_decompose_simple(
        model, input_ids, layer_idx, key_position
    )
    
    print(f"\nTop 3 vocabulary projections at position {key_position}:")
    for i, name in enumerate(token_names):
        print(f"  {i+1}. '{name}'")
    
    # Compute attention for each representation as KEY
    attention_patterns = {}
    
    for rep_name, rep_vector in decomposed.items():
        # Skip 'original' for cleaner visualization
        if rep_name == 'original':
            continue
            
        attention = compute_attention_with_modified_keys(
            model, input_ids, layer_idx, head_idx, key_position, rep_vector, rep_name
        )
        attention_patterns[rep_name] = attention
    
    # Visualize with histograms
    visualize_key_attention_histograms(
        attention_patterns, 
        tokens, 
        token_names,
        layer_idx,
        head_idx,
        key_position
    )
    
    return attention_patterns, decomposed, token_names

def visualize_key_attention_histograms(
    attention_patterns,
    tokens,
    vocab_tokens,
    layer_idx,
    head_idx, 
    key_position
):
    """
    Create S histograms with 4 columns showing which tokens each position attends to.
    S = number of query positions (up to 8 for visualization).
    """
    num_positions = min(8, len(tokens))  # Show up to 8 positions
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # Colors for the 4 components (3 vocab + dark matter)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2C3E50']
    component_names = [f"'{vocab_tokens[0]}'", f"'{vocab_tokens[1]}'", f"'{vocab_tokens[2]}'", "Dark Matter"]
    
    for pos_idx in range(num_positions):
        ax = axes[pos_idx]
        
        # Get attention values for this position to each component type
        attention_values = []
        for key in ['word1', 'word2', 'word3', 'dark_matter']:
            if key in attention_patterns:
                attention_values.append(attention_patterns[key][pos_idx].item())
            else:
                attention_values.append(0)
        
        # Create bar chart
        x = np.arange(4)
        bars = ax.bar(x, attention_values, color=colors)
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(component_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Attention Weight', fontsize=9)
        ax.set_title(f"Pos {pos_idx}: '{tokens[pos_idx]}'", fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(0.5, max(attention_values) * 1.1) if attention_values else 0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, attention_values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for idx in range(num_positions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(
        f'Key Vector Implantation: Which Components Do Positions Attend To?\n'
        f'Layer {layer_idx}, Head {head_idx}, Key Position {key_position} ("{tokens[key_position]}")',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("Summary: Which component gets most attention?")
    print('-'*50)
    
    for pos_idx in range(num_positions):
        attention_values = []
        for key in ['word1', 'word2', 'word3', 'dark_matter']:
            if key in attention_patterns:
                attention_values.append(attention_patterns[key][pos_idx].item())
            else:
                attention_values.append(0)
        
        max_idx = np.argmax(attention_values)
        max_val = attention_values[max_idx]
        
        print(f"Position {pos_idx:2d} ('{tokens[pos_idx]:10s}'): → {component_names[max_idx]:15s} ({max_val:.3f})")

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    # Test text
    text = "Doc 1: The blue fox lives in the forest. It eats berries. Doc 2: The blue fox actually eats fish. Query The blue fox eats"
    
    # Analyze multiple layers and heads
    for layer_idx in [5, 7, 11]:  # Early, middle, late
        for head_idx in [1, 5, 10]:  # Sample different heads
            print(f"\n{'='*60}")
            analyze_decomposed_attention(
                model, tokenizer, text, layer_idx, head_idx
            )

if __name__ == "__main__":
    main()