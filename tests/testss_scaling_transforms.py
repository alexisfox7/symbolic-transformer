import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_token_through_first_layer(model, token_id, position_idx, seq_length, device='cuda'):
    """Pass a single token through the first transformer layer at a specific position."""
    pad_token_id = 50256
    dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=device)
    dummy_input[0, position_idx] = token_id
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(dummy_input)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        
        # Initial hidden state
        hidden_states = inputs_embeds + position_embeds
        
        # Pass through first transformer block
        first_block = model.transformer.h[0]
        
        # Create attention mask
        attention_mask = torch.zeros((1, seq_length), device=device)
        attention_mask[0, position_idx] = 1.0
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through first layer
        outputs = first_block(hidden_states, attention_mask=extended_attention_mask)
        first_layer_output = outputs[0]
        
        return first_layer_output[0, position_idx, :]

def apply_scaling_transform(raw_embeddings, target_residual, method='magnitude_match'):
    """Apply various scaling transformations to raw embeddings."""
    
    if method == 'raw':
        return raw_embeddings
    
    elif method == 'magnitude_match':
        # Scale to match target residual magnitude
        raw_norms = torch.norm(raw_embeddings, dim=1, keepdim=True)
        target_norm = torch.norm(target_residual)
        avg_raw_norm = torch.mean(raw_norms)
        scale_factor = target_norm / avg_raw_norm
        return raw_embeddings * scale_factor
    
    elif method == 'unit_normalize':
        # Normalize to unit vectors then scale to target magnitude
        normalized = F.normalize(raw_embeddings, p=2, dim=1)
        target_norm = torch.norm(target_residual)
        return normalized * target_norm
    
    elif method == 'layernorm_scale':
        # Apply layer normalization style scaling
        mean = raw_embeddings.mean(dim=1, keepdim=True)
        std = raw_embeddings.std(dim=1, keepdim=True)
        normalized = (raw_embeddings - mean) / (std + 1e-5)
        # Scale to match target residual magnitude
        target_norm = torch.norm(target_residual)
        current_norm = torch.norm(normalized, dim=1).mean()
        return normalized * (target_norm / current_norm)
    
    elif method == 'sqrt_d_scale':
        # Scale by sqrt(d_model) like in transformers
        d_model = raw_embeddings.shape[1]
        return raw_embeddings * (d_model ** 0.5)
    
    elif method == 'learned_scale':
        # Learn a single scale factor per token position
        raw_norms = torch.norm(raw_embeddings, dim=1, keepdim=True)
        target_norm = torch.norm(target_residual)
        # Simple heuristic: scale based on position in top-k
        scales = torch.tensor([1.2, 1.0, 0.8], device=raw_embeddings.device).unsqueeze(1)
        scaled_embeddings = raw_embeddings * scales
        # Then magnitude match
        scaled_norms = torch.norm(scaled_embeddings, dim=1, keepdim=True)
        avg_scaled_norm = torch.mean(scaled_norms)
        final_scale = target_norm / avg_scaled_norm
        return scaled_embeddings * final_scale
    
    elif method == 'residual_stats_match':
        # Match the statistical properties of the target residual
        target_mean = target_residual.mean()
        target_std = target_residual.std()
        
        # Apply to each embedding
        transformed = []
        for emb in raw_embeddings:
            emb_mean = emb.mean()
            emb_std = emb.std()
            # Standardize then scale to match target stats
            standardized = (emb - emb_mean) / (emb_std + 1e-8)
            matched = standardized * target_std + target_mean
            transformed.append(matched)
        
        return torch.stack(transformed)
    
    else:
        raise ValueError(f"Unknown scaling method: {method}")

def get_residual_and_decompose_with_scaling(model, input_ids, layer_idx, position_idx, scaling_method='magnitude_match'):
    """Get residual stream and decompose using scaling transformations."""
    device = input_ids.device
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get residual
        if layer_idx == -1:
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
    
    # Apply layer normalization for LogitLens
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    # Get top 3 tokens
    top3_values, top3_indices = torch.topk(logits[0], k=3)
    
    # Get raw embeddings
    raw_embeddings = model.transformer.wte(top3_indices)
    
    # Apply scaling transformation
    scaled_embeddings = apply_scaling_transform(raw_embeddings, residual_at_pos, scaling_method)
    
    # Use least squares to find coefficients
    A = scaled_embeddings.T  # Shape: [hidden_dim, 3]
    b = residual_at_pos      # Shape: [hidden_dim]
    
    # Solve least squares: A @ coeffs = b
    coeffs, residuals, rank, s = torch.linalg.lstsq(A, b)
    
    # Reconstruct using the coefficients
    reconstruction = torch.matmul(A, coeffs)
    dark_matter = residual_at_pos - reconstruction
    
    # Compute reconstruction error
    error = torch.norm(dark_matter).item()
    error_pct = 100 * error / torch.norm(residual_at_pos).item()
    
    # Compute cosine similarities
    similarities = {}
    for i, embedding in enumerate(scaled_embeddings):
        similarity = F.cosine_similarity(
            residual_at_pos.unsqueeze(0), 
            embedding.unsqueeze(0)
        ).item()
        similarities[f'word{i+1}'] = similarity
    
    # Store coefficients
    coefficients = {f'coeff{i+1}': coeffs[i].item() for i in range(3)}
    
    decomposed = {
        'original': residual_at_pos,
        'word1': coeffs[0] * scaled_embeddings[0],
        'word2': coeffs[1] * scaled_embeddings[1], 
        'word3': coeffs[2] * scaled_embeddings[2],
        'dark_matter': dark_matter
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    # Return embedding norms for analysis
    raw_norms = [torch.norm(raw_embeddings[i]).item() for i in range(3)]
    scaled_norms = [torch.norm(scaled_embeddings[i]).item() for i in range(3)]
    
    return {
        'decomposed': decomposed,
        'token_names': token_names,
        'similarities': similarities,
        'coefficients': coefficients,
        'error_pct': error_pct,
        'raw_norms': raw_norms,
        'scaled_norms': scaled_norms,
        'target_norm': torch.norm(residual_at_pos).item(),
        'scaling_method': scaling_method
    }

def compare_scaling_methods(model, input_ids, layer_idx, position_idx):
    """Compare different scaling methods on the same residual."""
    
    scaling_methods = [
        'raw',
        'magnitude_match', 
        'unit_normalize',
        'layernorm_scale',
        'sqrt_d_scale',
        'learned_scale',
        'residual_stats_match'
    ]
    
    # Also compare with first-layer transformation
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
        residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        top3_values, top3_indices = torch.topk(logits[0], k=3)
        
        # First-layer transformed embeddings
        first_layer_embeddings = []
        for token_id in top3_indices:
            transformed = get_token_through_first_layer(
                model, token_id.item(), position_idx, input_ids.shape[1], input_ids.device
            )
            first_layer_embeddings.append(transformed)
        first_layer_embeddings = torch.stack(first_layer_embeddings)
        
        # Reconstruct with first-layer
        A_first = first_layer_embeddings.T
        b = residual_at_pos
        coeffs_first = torch.linalg.lstsq(A_first, b).solution
        reconstruction_first = torch.matmul(A_first, coeffs_first)
        error_first = torch.norm(residual_at_pos - reconstruction_first).item()
        error_first_pct = 100 * error_first / torch.norm(residual_at_pos).item()
        first_layer_norms = [torch.norm(first_layer_embeddings[i]).item() for i in range(3)]
    
    print(f"\nComparing scaling methods for layer {layer_idx}, position {position_idx}:")
    print("-" * 80)
    
    results = {}
    
    for method in scaling_methods:
        result = get_residual_and_decompose_with_scaling(
            model, input_ids, layer_idx, position_idx, method
        )
        results[method] = result
        
        print(f"\n{method.upper().replace('_', ' ')} METHOD:")
        print(f"  Tokens: {result['token_names']}")
        print(f"  Reconstruction error: {result['error_pct']:.1f}%")
        print(f"  Coefficients: {[f'{coeff:.3f}' for coeff in result['coefficients'].values()]}")
        print(f"  Cosine similarities: {[f'{sim:.3f}' for sim in result['similarities'].values()]}")
        print(f"  Raw norms: {[f'{norm:.1f}' for norm in result['raw_norms']]}")
        print(f"  Scaled norms: {[f'{norm:.1f}' for norm in result['scaled_norms']]}")
        print(f"  Target norm: {result['target_norm']:.1f}")
    
    # Show first-layer comparison
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    print(f"\nFIRST-LAYER TRANSFORMATION (comparison):")
    print(f"  Tokens: {token_names}")
    print(f"  Reconstruction error: {error_first_pct:.1f}%")
    print(f"  Transformed norms: {[f'{norm:.1f}' for norm in first_layer_norms]}")
    
    # Show improvements over raw
    raw_error = results['raw']['error_pct']
    print(f"\nIMPROVEMENT ANALYSIS (vs raw embeddings):")
    for method in scaling_methods[1:]:  # Skip 'raw'
        improvement = raw_error - results[method]['error_pct']
        print(f"  {method.replace('_', ' ').title()}: {improvement:+.1f} percentage points")
    
    first_improvement = raw_error - error_first_pct
    print(f"  First-layer transform: {first_improvement:+.1f} percentage points")
    
    return results

def visualize_scaling_comparison(results_by_layer, save_path='scaling_methods_comparison.png'):
    """Visualize comparison of scaling methods across layers."""
    
    methods = ['raw', 'magnitude_match', 'unit_normalize', 'layernorm_scale', 
               'sqrt_d_scale', 'learned_scale', 'residual_stats_match']
    
    layers = list(results_by_layer.keys())
    n_methods = len(methods)
    
    # Collect errors for each method across layers
    errors_by_method = {method: [] for method in methods}
    
    for layer in layers:
        for method in methods:
            error = results_by_layer[layer][method]['error_pct']
            errors_by_method[method].append(error)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(methods):
        plt.plot(layers, errors_by_method[method], 'o-', 
                label=method.replace('_', ' ').title(), 
                linewidth=2, markersize=6, color=colors[i])
    
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Reconstruction Error (%)', fontsize=14)
    plt.title('Scaling Method Comparison Across Layers', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    print("=" * 80)
    print("SCALING TRANSFORMATION METHODS COMPARISON")
    print("=" * 80)
    
    # Test text
    test_text = "The cat sat on the mat"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"Test text: '{test_text}'")
    print(f"Tokens: {tokens}")
    
    # Test across different layers and positions
    results_by_layer = {}
    
    for layer_idx in [1, 2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        
        # Test on middle position
        position_idx = len(tokens) // 2
        
        results = compare_scaling_methods(model, input_ids, layer_idx, position_idx)
        results_by_layer[layer_idx] = results
    
    # Visualize results
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION...")
    
    save_path = visualize_scaling_comparison(results_by_layer)
    print(f"Comparison plot saved as: {save_path}")
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print("=" * 60)
    
    # Find best method on average
    methods = ['magnitude_match', 'unit_normalize', 'layernorm_scale', 
               'sqrt_d_scale', 'learned_scale', 'residual_stats_match']
    
    avg_errors = {}
    for method in methods:
        errors = [results_by_layer[layer][method]['error_pct'] for layer in results_by_layer.keys()]
        avg_errors[method] = np.mean(errors)
    
    # Sort by performance
    sorted_methods = sorted(avg_errors.items(), key=lambda x: x[1])
    
    print(f"Average reconstruction error across all tested layers/positions:")
    for method, avg_error in sorted_methods:
        raw_avg = np.mean([results_by_layer[layer]['raw']['error_pct'] for layer in results_by_layer.keys()])
        improvement = raw_avg - avg_error
        print(f"  {method.replace('_', ' ').title():<20}: {avg_error:>6.1f}% (improvement: {improvement:+.1f}pp)")
    
    print(f"\nBest scaling method: {sorted_methods[0][0].replace('_', ' ').title()}")

if __name__ == "__main__":
    main()