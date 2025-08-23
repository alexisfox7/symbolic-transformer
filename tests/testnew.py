import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

def get_token_through_first_layer(model, token_id, position_idx, seq_length, device):
    """Get a token's representation after passing through the first layer."""
    with torch.no_grad():
        # Create a dummy input with the target token at the specified position
        dummy_input = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        dummy_input[0, position_idx] = token_id
        
        # Get embeddings
        token_embeds = model.transformer.wte(dummy_input)
        position_embeds = model.transformer.wpe(torch.arange(seq_length, device=device))
        hidden_states = token_embeds + position_embeds
        
        # Pass through first layer
        first_layer = model.transformer.h[0]
        hidden_states = first_layer(hidden_states)[0]
        
        # Extract representation at the position
        return hidden_states[0, position_idx, :]

def decompose_with_least_squares(model, input_ids, layer_idx, position_idx):
    """
    Decompose representation using least squares to find optimal coefficients
    for top 3 tokens, with dark matter as the reconstruction residual.
    
    Args:
        model: GPT2 model
        input_ids: Input token IDs
        layer_idx: Layer to analyze (-1 for embeddings, 0 for after layer 0, etc.)
        position_idx: Position in sequence to decompose
    
    Returns:
        decomposed: Dict with original, word1, word2, word3, dark_matter, and coefficients
        token_names: List of token names
        reconstruction_error: L2 norm of dark matter
    """
    device = input_ids.device
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get representation at specified layer and position
        if layer_idx == -1:
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
    
    # Get top 3 tokens using LogitLens
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    top3_values, top3_indices = torch.topk(logits[0], k=3)
    
    # Get first-layer transformed embeddings for the top 3 tokens
    top3_embeddings = []
    for token_id in top3_indices:
        transformed = get_token_through_first_layer(
            model, token_id.item(), position_idx, seq_length, device
        )
        top3_embeddings.append(transformed)
    
    # Stack embeddings into matrix A: [hidden_dim, 3]
    A = torch.stack(top3_embeddings, dim=1)
    
    # Target vector b: [hidden_dim]
    b = residual_at_pos
    
    # Solve least squares: find coefficients x such that Ax ≈ b
    # Using torch.linalg.lstsq for numerical stability
    coefficients, residuals, rank, s = torch.linalg.lstsq(A, b.unsqueeze(1))
    coefficients = coefficients.squeeze()
    
    # Reconstruct using the coefficients
    reconstruction = torch.matmul(A, coefficients.unsqueeze(1)).squeeze()
    
    # Dark matter is the residual
    dark_matter = b - reconstruction
    
    # Prepare output dictionary
    decomposed = {
        'original': residual_at_pos,
        'word1': top3_embeddings[0],
        'word2': top3_embeddings[1],
        'word3': top3_embeddings[2],
        'coefficients': coefficients,
        'reconstruction': reconstruction,
        'dark_matter': dark_matter
    }
    
    # Get token names
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    # Calculate reconstruction metrics
    reconstruction_error = torch.norm(dark_matter).item()
    relative_error = reconstruction_error / torch.norm(residual_at_pos).item()
    
    # Calculate variance explained (R²)
    total_variance = torch.var(residual_at_pos).item()
    residual_variance = torch.var(dark_matter).item()
    r_squared = 1 - (residual_variance / total_variance) if total_variance > 0 else 0
    
    metrics = {
        'reconstruction_error': reconstruction_error,
        'relative_error': relative_error,
        'r_squared': r_squared,
        'dark_matter_norm': torch.norm(dark_matter).item(),
        'original_norm': torch.norm(residual_at_pos).item()
    }
    
    return decomposed, token_names, metrics

def compute_attention_with_decomposed_keys(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    key_position
):
    """
    Compute how query at one position attends to decomposed keys at another position,
    using least squares decomposition.
    """
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
        
        # Get decomposed representations at key position using least squares
        decomposed, token_names, metrics = decompose_with_least_squares(
            model, input_ids, layer_idx-1, key_position
        )
        
        # Compute key vectors for each decomposed component
        attention_scores = {}
        
        for rep_name in ['word1', 'word2', 'word3', 'reconstruction', 'dark_matter']:
            if rep_name == 'reconstruction':
                # Reconstruct using coefficients
                rep_vector = decomposed['coefficients'][0] * decomposed['word1'] + \
                            decomposed['coefficients'][1] * decomposed['word2'] + \
                            decomposed['coefficients'][2] * decomposed['word3']
            else:
                rep_vector = decomposed[rep_name]
            
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
        
        # Also compute the weighted attention based on coefficients
        weighted_attention = (
            decomposed['coefficients'][0] * attention_scores['word1'] +
            decomposed['coefficients'][1] * attention_scores['word2'] +
            decomposed['coefficients'][2] * attention_scores['word3'] +
            attention_scores['dark_matter']
        )
        
        attention_scores['weighted_total'] = weighted_attention.item()
    
    return attention_scores, token_names, decomposed['coefficients'], metrics

def analyze_decomposition_quality(model, tokenizer, text, layer_idx=5):
    """Analyze the quality of least squares decomposition across all positions."""
    input_ids = tokenizer.encode(text, return_tensors='pt')
    seq_len = input_ids.shape[1]
    
    print(f"Analyzing text: '{text}'")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}\n")
    
    all_metrics = []
    
    for pos in range(seq_len):
        decomposed, token_names, metrics = decompose_with_least_squares(
            model, input_ids, layer_idx, pos
        )
        
        token_at_pos = tokenizer.decode([input_ids[0, pos].item()])
        print(f"Position {pos} ('{token_at_pos}'):")
        print(f"  Top 3 tokens: {token_names}")
        print(f"  Coefficients: {decomposed['coefficients'].cpu().numpy()}")
        print(f"  R²: {metrics['r_squared']:.4f}")
        print(f"  Relative error: {metrics['relative_error']:.4f}")
        print(f"  Dark matter norm: {metrics['dark_matter_norm']:.4f}")
        print()
        
        all_metrics.append(metrics)
    
    # Summary statistics
    avg_r2 = sum(m['r_squared'] for m in all_metrics) / len(all_metrics)
    avg_rel_error = sum(m['relative_error'] for m in all_metrics) / len(all_metrics)
    
    print(f"\nSummary:")
    print(f"  Average R²: {avg_r2:.4f}")
    print(f"  Average relative error: {avg_rel_error:.4f}")
    
    return all_metrics

# Example usage
if __name__ == "__main__":
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    text = "The cat sat on the mat"
    
    # Analyze decomposition quality
    print("=" * 60)
    print("DECOMPOSITION QUALITY ANALYSIS")
    print("=" * 60)
    metrics = analyze_decomposition_quality(model, tokenizer, text, layer_idx=5)
    
    # Analyze attention with decomposed keys
    print("\n" + "=" * 60)
    print("ATTENTION TO DECOMPOSED COMPONENTS")
    print("=" * 60)
    
    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_scores, token_names, coeffs, metrics = compute_attention_with_decomposed_keys(
        model, input_ids, 
        layer_idx=5, head_idx=0,
        query_position=4,  # "the"
        key_position=2     # "sat"
    )
    
    print(f"\nQuery position 4 attending to decomposed key at position 2:")
    print(f"Top 3 tokens at key position: {token_names}")
    print(f"Coefficients: {coeffs.cpu().numpy()}")
    print(f"\nAttention scores:")
    for name, score in attention_scores.items():
        print(f"  {name}: {score:.4f}")
    print(f"\nReconstruction metrics:")
    print(f"  R²: {metrics['r_squared']:.4f}")
    print(f"  Relative error: {metrics['relative_error']:.4f}")