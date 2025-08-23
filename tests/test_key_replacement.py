# test_key_replacement.py
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
    top3_values, top3_indices = torch.topk(logits[0], k=3)
    top3_embeddings = model.transformer.wte(top3_indices)
    
    # Compute cosine similarities with intermediate representation
    intermediate_norm = F.normalize(residual_at_pos.unsqueeze(0), p=2, dim=1)
    similarities = {}
    for i, embedding in enumerate(top3_embeddings):
        token_norm = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
        similarity = torch.cosine_similarity(intermediate_norm, token_norm, dim=1)
        similarities[f'word{i+1}'] = similarity.item()
    
    decomposed = {
        'original': residual_at_pos,
        'word1': top3_embeddings[0],
        'word2': top3_embeddings[1],
        'word3': top3_embeddings[2],
        'dark_matter': residual_at_pos - top3_embeddings.sum(dim=0)
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    return decomposed, token_names, similarities, top3_values

def compute_attention_with_modified_keys(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    key_replacement_components  # List of components to use for all key positions
):
    """
    Compute attention pattern when replacing ALL key vectors with specified components.
    Query vector at query_position remains as original "mush".
    """
    # Get the attention layer
    attn_layer = model.transformer.h[layer_idx].attn
    
    # First, get the original hidden states
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Apply layer norm before attention
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute original Q, K, V from the NORMED hidden states
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

        # Replace ALL key vectors with the decomposed components
        # Create modified key matrix by stacking the components
        modified_keys = torch.zeros_like(k[0, head_idx])  # [seq_len, head_dim]
        
        for pos in range(seq_len):
            # For each position, use one of the decomposed components cyclically
            component_idx = pos % len(key_replacement_components)
            component_name = list(key_replacement_components.keys())[component_idx]
            component_vector = key_replacement_components[component_name]
            
            # Apply layer norm to the component
            component_normed = model.transformer.h[layer_idx].ln_1(
                component_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project to key space
            component_qkv = attn_layer.c_attn(component_normed.unsqueeze(0).unsqueeze(0))
            component_k = component_qkv[:, :, hidden_dim:2*hidden_dim]  # Extract K part
            
            component_k = component_k.view(1, 1, num_heads, head_dim)
            modified_keys[pos, :] = component_k[0, 0, head_idx, :]
        
        # Replace all keys for this head
        k[0, head_idx] = modified_keys
        
        # Keep query as original (the "mush")
        # q remains unchanged
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Extract attention pattern for specific head and position
        attention_pattern = attention_weights[0, head_idx, query_position, :]
        
    return attention_pattern

def analyze_key_replacement_attention(model, tokenizer, text, layer_idx=5, head_idx=0):
    """
    Main analysis function for key replacement experiment.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer_idx}, Head {head_idx}")
    
    # Pick position to analyze (e.g., last token)
    query_position = len(tokens) - 1
    
    # Get decomposed representations from the query position
    decomposed, token_names, similarities, logit_values = get_residual_and_decompose_simple(
        model, input_ids, layer_idx, query_position
    )
    
    print(f"\nTop 3 vocabulary projections at position {query_position}:")
    for i, name in enumerate(token_names):
        logit_val = logit_values[i].item()
        print(f"  {i+1}. '{name}' (logit: {logit_val:.4f})")
    
    print(f"\nCosine similarities with intermediate representation:")
    for i, (key, sim) in enumerate(similarities.items()):
        token_name = token_names[i]
        print(f"  {key} ('{token_name}'): {sim:.4f}")
    
    # Test different key replacement strategies
    attention_patterns = {}
    
    # Original attention (no replacement)
    original_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {'original': decomposed['original']}  # Use original as all keys
    )
    attention_patterns['original'] = original_attention
    
    # Replace all keys with just word1
    word1_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {'word1': decomposed['word1']}
    )
    attention_patterns['all_keys_word1'] = word1_attention
    
    # Replace all keys with just word2
    word2_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {'word2': decomposed['word2']}
    )
    attention_patterns['all_keys_word2'] = word2_attention
    
    # Replace all keys with just word3
    word3_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {'word3': decomposed['word3']}
    )
    attention_patterns['all_keys_word3'] = word3_attention
    
    # Replace all keys with just dark matter
    dark_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {'dark_matter': decomposed['dark_matter']}
    )
    attention_patterns['all_keys_dark_matter'] = dark_attention
    
    # Replace keys cycling through all 4 components
    mixed_attention = compute_attention_with_modified_keys(
        model, input_ids, layer_idx, head_idx, query_position, 
        {
            'word1': decomposed['word1'],
            'word2': decomposed['word2'], 
            'word3': decomposed['word3'],
            'dark_matter': decomposed['dark_matter']
        }
    )
    attention_patterns['cycling_keys'] = mixed_attention
    
    # Visualize
    visualize_key_replacement_comparison(
        attention_patterns, 
        tokens, 
        token_names,
        layer_idx,
        head_idx,
        query_position
    )
    
    return attention_patterns, decomposed, token_names

def visualize_key_replacement_comparison(
    attention_patterns,
    tokens,
    vocab_tokens,
    layer_idx,
    head_idx, 
    query_position
):
    """
    Create comparison visualization for key replacement experiment.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot each attention pattern
    plot_configs = [
        ('original', 'Original (No Replacement)', axes[0, 0]),
        ('all_keys_word1', f"All Keys = Word1: '{vocab_tokens[0]}'", axes[0, 1]),
        ('all_keys_word2', f"All Keys = Word2: '{vocab_tokens[1]}'", axes[0, 2]),
        ('all_keys_word3', f"All Keys = Word3: '{vocab_tokens[2]}'", axes[1, 0]),
        ('all_keys_dark_matter', 'All Keys = Dark Matter', axes[1, 1]),
        ('cycling_keys', 'Keys Cycling Through All 4', axes[1, 2])
    ]
    
    vmax = max(pattern.max().item() for pattern in attention_patterns.values())
    
    for rep_name, title, ax in plot_configs:
        if rep_name in attention_patterns:
            attention = attention_patterns[rep_name].cpu().numpy()
            
            # Create bar plot
            x = np.arange(len(tokens))
            bars = ax.bar(x, attention, color='steelblue', alpha=0.7)
            
            # Highlight high attention
            threshold = attention.mean() + attention.std()
            for i, (bar, val) in enumerate(zip(bars, attention)):
                if val > threshold:
                    bar.set_color('darkred')
                    bar.set_alpha(0.9)
            
            ax.set_xticks(x)
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_ylabel('Attention Weight')
            ax.set_title(title)
            ax.set_ylim(0, vmax * 1.1)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Attention Patterns with Key Vector Replacement\n'
        f'Layer {layer_idx}, Head {head_idx}, Query Position {query_position} (Original "Mush")',
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    # Test text
    text = "The blue fox lives in the forest. It eats berries. The blue fox actually eats fish. The blue fox eats"

    # Analyze one specific layer/head to see results clearly
    layer_idx, head_idx = 5, 1
    print(f"\n{'='*80}")
    analyze_key_replacement_attention(
        model, tokenizer, text, layer_idx, head_idx
    )

if __name__ == "__main__":
    main()