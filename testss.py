import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def get_residual_and_decompose(model, input_ids, layer_idx, position_idx):
    """
    Get residual stream and decompose into top 3 tokens + dark matter.
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

def compute_attention_to_decomposed_keys(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    key_position
):
    """Compute how query at one position attends to decomposed keys at another position."""
    
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
        
        # Get Q from query position
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        query_vector = q[0, head_idx, query_position, :]
        
        # Get decomposed representations at key position
        decomposed, token_names, similarities, _ = get_residual_and_decompose(
            model, input_ids, layer_idx-1, key_position
        )
        
        # Compute key vectors for each decomposed representation
        attention_scores = {}
        
        for rep_name, rep_vector in decomposed.items():
            # Apply layer norm to decomposed vector
            rep_normed = model.transformer.h[layer_idx].ln_1(
                rep_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project to key space
            rep_kv = attn_layer.c_attn(rep_normed.unsqueeze(0).unsqueeze(0))
            rep_k = rep_kv[:, :, hidden_dim:2*hidden_dim]
            rep_k = rep_k.view(1, 1, num_heads, head_dim)[0, 0, head_idx, :]
            
            # Compute attention score (before softmax)
            score = torch.dot(query_vector, rep_k) / (head_dim ** 0.5)
            attention_scores[rep_name] = score.item()
    
    return attention_scores, token_names, similarities

def analyze_decomposed_key_attention(model, tokenizer, text, layer_idx=5, head_idx=0):
    """Analyze how queries attend to decomposed keys."""
    
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer_idx}, Head {head_idx}")
    
    # Pick query position (last token)
    query_position = len(tokens) - 1
    print(f"\nQuery position: {query_position} ('{tokens[query_position]}')")
    
    # Analyze attention to each key position
    all_attention_scores = []
    all_token_names = []
    
    for key_pos in range(len(tokens)):
        scores, token_names, similarities = compute_attention_to_decomposed_keys(
            model, input_ids, layer_idx, head_idx, query_position, key_pos
        )
        all_attention_scores.append(scores)
        all_token_names.append(token_names)
        
        print(f"\nKey position {key_pos} ('{tokens[key_pos]}'):")
        print(f"  Top 3 projections: {token_names}")
        print(f"  Cosine similarities:")
        for i, (key, sim) in enumerate(similarities.items()):
            token_name = token_names[i]
            print(f"    {key} ('{token_name}'): {sim:.4f}")
        print(f"  Attention scores:")
        for rep_name, score in scores.items():
            if rep_name.startswith('word'):
                idx = int(rep_name[-1]) - 1
                print(f"    {rep_name} ('{token_names[idx]}'): {score:.4f}")
            else:
                print(f"    {rep_name}: {score:.4f}")
    
    # Visualize
    visualize_decomposed_key_attention(
        all_attention_scores, tokens, all_token_names, 
        layer_idx, head_idx, query_position
    )
    
    return all_attention_scores, all_token_names

def visualize_decomposed_key_attention(
    all_attention_scores, tokens, all_token_names,
    layer_idx, head_idx, query_position
):
    """Visualize attention from query to decomposed keys."""
    
    n_positions = len(tokens)
    rep_types = ['original', 'word1', 'word2', 'word3', 'dark_matter']
    
    # Create matrix of attention scores
    attention_matrix = np.zeros((len(rep_types), n_positions))
    for pos, scores in enumerate(all_attention_scores):
        for i, rep_type in enumerate(rep_types):
            attention_matrix[i, pos] = scores[rep_type]
    
    # Apply softmax across all decomposed keys for each position
    # This shows relative attention within each position
    attention_softmax = np.zeros_like(attention_matrix)
    for pos in range(n_positions):
        scores = attention_matrix[:, pos]
        exp_scores = np.exp(scores - scores.max())
        attention_softmax[:, pos] = exp_scores / exp_scores.sum()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Raw attention scores
    im1 = ax1.imshow(attention_matrix, aspect='auto', cmap='coolwarm')
    ax1.set_yticks(range(len(rep_types)))
    ax1.set_yticklabels(rep_types)
    ax1.set_xticks(range(n_positions))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_title(f'Raw Attention Scores from Query "{tokens[query_position]}" to Decomposed Keys')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Representation Type')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations for high values
    for i in range(len(rep_types)):
        for j in range(n_positions):
            val = attention_matrix[i, j]
            if abs(val) > np.percentile(np.abs(attention_matrix), 75):
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val < 0 else 'black', fontsize=8)
    
    # Plot 2: Softmax normalized (shows relative importance within each position)
    im2 = ax2.imshow(attention_softmax, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_yticks(range(len(rep_types)))
    ax2.set_yticklabels(rep_types)
    ax2.set_xticks(range(n_positions))
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.set_title('Softmax Normalized Attention (Relative Within Each Position)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Representation Type')
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations for high values
    for i in range(len(rep_types)):
        for j in range(n_positions):
            val = attention_softmax[i, j]
            if val > 0.3:  # Highlight significant attention
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if val > 0.5 else 'black', fontsize=8)
    
    plt.suptitle(
        f'Attention to Decomposed Keys\nLayer {layer_idx}, Head {head_idx}, Query Position {query_position}',
        fontsize=14
    )
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Stacked bar chart showing decomposition
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    
    bottom = np.zeros(n_positions)
    colors = ['blue', 'green', 'orange', 'red', 'gray']
    
    for i, rep_type in enumerate(rep_types):
        ax3.bar(range(n_positions), attention_softmax[i, :], 
               bottom=bottom, label=rep_type, color=colors[i], alpha=0.8)
        bottom += attention_softmax[i, :]
    
    ax3.set_xticks(range(n_positions))
    ax3.set_xticklabels(tokens, rotation=45, ha='right')
    ax3.set_ylabel('Attention Weight')
    ax3.set_title(f'Stacked Attention Distribution from "{tokens[query_position]}" to Decomposed Keys')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
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
    text = "The cat sat on the mat"
    
    # Analyze different layers and heads
    for layer_idx in [2, 5, 8]:
        for head_idx in [0, 3, 7]:
            print(f"\n{'='*60}")
            analyze_decomposed_key_attention(
                model, tokenizer, text, layer_idx, head_idx
            )

if __name__ == "__main__":
    main()