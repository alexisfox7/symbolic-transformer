# gpt2_decomposed_attention.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from typing import Dict, List, Tuple

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
            # Before layer layer_idx (so index is layer_idx, which is after layer_idx-1)
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
    
    # Apply layer normalization for LogitLens (all layers need ln_f for vocab projection)
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    # Get top 3
    top3_values, top3_indices = torch.topk(logits[0], k=3)
    
    # Use first-layer transformed embeddings instead of raw embeddings
    top3_embeddings = []
    for token_id in top3_indices:
        transformed = get_token_through_first_layer(
            model, token_id.item(), position_idx, seq_length, device
        )
        top3_embeddings.append(transformed)
    top3_embeddings = torch.stack(top3_embeddings)
    
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

def compute_cosine_similarities(decomposed_dict, intermediate_rep):
    """
    Compute cosine similarities between top tokens and intermediate representation.
    
    Args:
        decomposed_dict: Dictionary with token embeddings ('word1', 'word2', 'word3')
        intermediate_rep: The intermediate representation vector to compare against
    
    Returns:
        Dictionary of cosine similarities
    """
    similarities = {}
    
    # Normalize intermediate representation
    intermediate_norm = F.normalize(intermediate_rep.unsqueeze(0), p=2, dim=1)
    
    for key in ['word1', 'word2', 'word3']:
        if key in decomposed_dict:
            # Normalize token embedding
            token_norm = F.normalize(decomposed_dict[key].unsqueeze(0), p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(intermediate_norm, token_norm, dim=1)
            similarities[key] = similarity.item()
    
    return similarities

def compute_attention_with_modified_query(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    modified_query_vector,
    decomposition_type  # Add this to know what type we're using
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
        qkv = attn_layer.c_attn(hidden_states_normed)  # This is what actually happens
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
        
        # If decomposition_type == 'original', we keep the original Q unchanged
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Extract attention pattern for specific head and position
        attention_pattern = attention_weights[0, head_idx, query_position, :]
        
    return attention_pattern

def analyze_decomposed_attention(model, tokenizer, text, layer_idx=5, head_idx=0):
    """
    Main analysis function.
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
    
    # Get decomposed representations
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
    
    # Compute attention for each representation
    attention_patterns = {}
    
    for rep_name, rep_vector in decomposed.items():
        attention = compute_attention_with_modified_query(
            model, input_ids, layer_idx, head_idx, query_position, rep_vector, rep_name
        )
        attention_patterns[rep_name] = attention
    
    # Visualize
    visualize_attention_comparison(
        attention_patterns, 
        tokens, 
        token_names,
        layer_idx,
        head_idx,
        query_position
    )
    
    return attention_patterns, decomposed, token_names

def visualize_attention_comparison(
    attention_patterns,
    tokens,
    vocab_tokens,
    layer_idx,
    head_idx, 
    query_position
):
    """
    Create comparison visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot each attention pattern
    plot_configs = [
        ('original', 'Original', axes[0, 0]),
        ('word1', f"Word 1: '{vocab_tokens[0]}'", axes[0, 1]),
        ('word2', f"Word 2: '{vocab_tokens[1]}'", axes[0, 2]),
        ('word3', f"Word 3: '{vocab_tokens[2]}'", axes[1, 0]),
        ('dark_matter', 'Dark Matter', axes[1, 1])
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
    
    # Difference plot in last subplot
    ax_diff = axes[1, 2]
    original = attention_patterns['original'].cpu().numpy()
    
    # Calculate variance across decomposed patterns
    decomposed_patterns = np.stack([
        attention_patterns[k].cpu().numpy() 
        for k in ['word1', 'word2', 'word3', 'dark_matter']
    ])
    variance = decomposed_patterns.var(axis=0)
    
    x = np.arange(len(tokens))
    ax_diff.bar(x, variance, color='purple', alpha=0.7)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(tokens, rotation=45, ha='right')
    ax_diff.set_ylabel('Variance')
    ax_diff.set_title('Attention Variance Across Decompositions')
    ax_diff.grid(True, alpha=0.3)
    
    plt.suptitle(
        f'Attention Patterns with Decomposed Queries (Using First Layer Embeddings)\n'
        f'Layer {layer_idx}, Head {head_idx}, Query Position {query_position}',
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

    # Analyze multiple layers and heads
    for layer_idx in [0, 7, 11]:  # Early, middle, late
        for head_idx in [1, 4, 10]:  # Sample different heads
            print(f"\n{'='*60}")
            analyze_decomposed_attention(
                model, tokenizer, text, layer_idx, head_idx
            )

if __name__ == "__main__":
    main()