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

def compute_attention_with_modified_keys_for_query(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    key_position,
    modified_key_vector,
    decomposition_type  # Add this to know what type we're using
):
    """
    Compute how much query_position attends to a modified key at key_position.
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
        
        # CRUCIAL: Apply layer norm before attention (this is what GPT-2 does!)
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute Q, K, V from the NORMED hidden states
        qkv = attn_layer.c_attn(hidden_states_normed)  # This is what actually happens
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Now handle the modification
        if decomposition_type != 'original':
            # For decomposed vectors, we need to apply layer norm too!
            modified_vector_normed = model.transformer.h[layer_idx].ln_1(
                modified_key_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project modified vector to QKV space
            modified_qkv = attn_layer.c_attn(modified_vector_normed.unsqueeze(0).unsqueeze(0))
            modified_k = modified_qkv[:, :, hidden_dim:2*hidden_dim]
            
            modified_k = modified_k.view(1, 1, num_heads, head_dim).transpose(1, 2)
            
            # Replace K at the key position
            k[:, :, key_position, :] = modified_k[0, :, 0, :]
        
        # If decomposition_type == 'original', we keep the original K unchanged
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Extract how much query_position attends to key_position
        attention_value = attention_weights[0, head_idx, query_position, key_position].item()
        
    return attention_value

def analyze_what_position_attends_to(model, tokenizer, text, layer_idx=5, head_idx=0, query_position=None):
    """
    For a single query position, analyze what it attends to in ALL previous positions.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    # Default to last position as query
    if query_position is None:
        query_position = len(tokens) - 1
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer_idx}, Head {head_idx}")
    print(f"Query Position: {query_position} ('{tokens[query_position]}')")
    
    # For each position before query_position, decompose and test
    all_attention_data = []
    
    for key_pos in range(query_position):  # Only look at positions before query
        # Decompose this position
        decomposed, token_names = get_residual_and_decompose_simple(
            model, input_ids, layer_idx, key_pos
        )
        
        # Test attention to each component
        attention_to_components = {}
        
        for comp_name, comp_vector in decomposed.items():
            if comp_name == 'original':
                continue
                
            attention_val = compute_attention_with_modified_keys_for_query(
                model, input_ids, layer_idx, head_idx, 
                query_position, key_pos, comp_vector, comp_name
            )
            attention_to_components[comp_name] = attention_val
        
        all_attention_data.append({
            'position': key_pos,
            'token': tokens[key_pos],
            'top3_vocab': token_names,
            'attention_values': attention_to_components
        })
    
    # Visualize
    visualize_query_attention_patterns(
        all_attention_data,
        tokens,
        query_position,
        layer_idx,
        head_idx
    )
    
    return all_attention_data

def visualize_query_attention_patterns(
    attention_data,
    tokens,
    query_position,
    layer_idx,
    head_idx
):
    """
    Create histograms showing what query_position attends to at each previous position.
    """
    num_positions = len(attention_data)
    
    # Create grid - up to 8 histograms
    max_to_show = min(8, num_positions)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    # Colors for the 4 components
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#2C3E50']
    
    for idx in range(max_to_show):
        ax = axes[idx]
        data = attention_data[idx]
        
        # Get the vocabulary words for this position
        vocab_labels = [f"'{w}'" for w in data['top3_vocab']] + ["Dark Matter"]
        
        # Get attention values
        attention_vals = [
            data['attention_values'].get('word1', 0),
            data['attention_values'].get('word2', 0),
            data['attention_values'].get('word3', 0),
            data['attention_values'].get('dark_matter', 0)
        ]
        
        # Create bar chart
        x = np.arange(4)
        bars = ax.bar(x, attention_vals, color=colors)
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(vocab_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Attention Weight', fontsize=9)
        ax.set_title(f"Pos {data['position']}: '{data['token']}'", fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(0.1, max(attention_vals) * 1.1) if attention_vals else 0.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, attention_vals):
            if val > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        
        # Highlight the maximum
        max_idx = np.argmax(attention_vals)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)
    
    # Hide unused subplots
    for idx in range(max_to_show, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(
        f'What Does Position {query_position} ("{tokens[query_position]}") Attend To?\n'
        f'Layer {layer_idx}, Head {head_idx} - Decomposition of Positions 0-{min(7, query_position-1)}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"Summary: Position {query_position} ('{tokens[query_position]}') attention patterns")
    print('-'*60)
    
    total_vocab_attention = 0
    total_dark_attention = 0
    
    for data in attention_data:
        attention_vals = [
            data['attention_values'].get('word1', 0),
            data['attention_values'].get('word2', 0),
            data['attention_values'].get('word3', 0),
            data['attention_values'].get('dark_matter', 0)
        ]
        
        vocab_labels = data['top3_vocab'] + ["Dark Matter"]
        max_idx = np.argmax(attention_vals)
        max_val = attention_vals[max_idx]
        
        # Track totals
        total_vocab_attention += sum(attention_vals[:3])
        total_dark_attention += attention_vals[3]
        
        print(f"Pos {data['position']:2d} ('{data['token']:10s}'): → {vocab_labels[max_idx]:15s} ({max_val:.3f})")
    
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    print(f"Total attention to vocabulary components: {total_vocab_attention:.3f}")
    print(f"Total attention to dark matter: {total_dark_attention:.3f}")
    print(f"Ratio (vocab/dark): {total_vocab_attention/max(total_dark_attention, 0.001):.2f}")

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
            analyze_what_position_attends_to(
                model, tokenizer, text, layer_idx, head_idx
            )

if __name__ == "__main__":
    main()