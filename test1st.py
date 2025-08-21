import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

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


def decompose_with_first_layer(model, input_ids, layer_idx, position_idx, k=3):
    """
    Decompose residual using tokens transformed through the first layer.
    """
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual
        if layer_idx == -1:
            residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via LogitLens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        topk_values, topk_indices = torch.topk(logits, k=k)
        
        # Method 1: Raw embeddings (original approach)
        raw_embeddings = model.transformer.wte(topk_indices)
        
        # Method 2: First-layer transformed tokens
        transformed_embeddings = []
        for token_id in topk_indices:
            transformed = get_token_through_first_layer(
                model, token_id, position_idx, seq_length, device
            )
            transformed_embeddings.append(transformed)
        transformed_embeddings = torch.stack(transformed_embeddings)
        
        # Compare both approaches
        results = {
            'residual': residual,
            'topk_tokens': [tokenizer.decode([idx.item()]) for idx in topk_indices],
            'topk_indices': topk_indices,
            'topk_logits': topk_values,
            'raw_embeddings': raw_embeddings,
            'transformed_embeddings': transformed_embeddings,
        }
        
        # Compute reconstructions using least squares for both
        # Raw embeddings
        A_raw = raw_embeddings.T
        b = residual.unsqueeze(1)
        coeffs_raw = torch.linalg.lstsq(A_raw, b).solution.squeeze()
        recon_raw = (A_raw @ coeffs_raw.unsqueeze(1)).squeeze()
        error_raw = torch.norm(residual - recon_raw).item()
        
        # Transformed embeddings
        A_trans = transformed_embeddings.T
        coeffs_trans = torch.linalg.lstsq(A_trans, b).solution.squeeze()
        recon_trans = (A_trans @ coeffs_trans.unsqueeze(1)).squeeze()
        error_trans = torch.norm(residual - recon_trans).item()
        
        results['coeffs_raw'] = coeffs_raw
        results['coeffs_transformed'] = coeffs_trans
        results['reconstruction_raw'] = recon_raw
        results['reconstruction_transformed'] = recon_trans
        results['error_raw'] = error_raw
        results['error_transformed'] = error_trans
        
        # Compute cosine similarities
        for i in range(k):
            # Raw embedding similarity
            cos_sim_raw = F.cosine_similarity(
                residual.unsqueeze(0),
                raw_embeddings[i].unsqueeze(0)
            ).item()
            
            # Transformed embedding similarity
            cos_sim_trans = F.cosine_similarity(
                residual.unsqueeze(0),
                transformed_embeddings[i].unsqueeze(0)
            ).item()
            
            results[f'cosine_raw_{i}'] = cos_sim_raw
            results[f'cosine_transformed_{i}'] = cos_sim_trans
        
        return results


def simple_logit_lens_decomposition(model, input_ids, layer_idx, position_idx, k=3):
    """
    Simple top-k logit lens decomposition using first-layer transformed embeddings.
    Uses the same transformation as RVQ method but with simpler token selection.
    """
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual
        if layer_idx == -1:
            residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via LogitLens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        topk_values, topk_indices = torch.topk(logits, k=k)
        
        # Use first-layer transformed embeddings (for fair comparison)
        transformed_embeddings = []
        for token_id in topk_indices:
            transformed = get_token_through_first_layer(
                model, token_id, position_idx, seq_length, device
            )
            transformed_embeddings.append(transformed)
        transformed_embeddings = torch.stack(transformed_embeddings)
        
        # Compute reconstruction using least squares
        A = transformed_embeddings.T
        b = residual.unsqueeze(1)
        coeffs = torch.linalg.lstsq(A, b).solution.squeeze()
        reconstruction = (A @ coeffs.unsqueeze(1)).squeeze()
        error = torch.norm(residual - reconstruction).item()
        
        return {
            'residual': residual,
            'topk_tokens': [tokenizer.decode([idx.item()]) for idx in topk_indices],
            'topk_indices': topk_indices,
            'topk_logits': topk_values,
            'coefficients': coeffs,
            'reconstruction': reconstruction,
            'error': error,
            'embeddings': transformed_embeddings
        }


def true_rvq_decomposition(model, input_ids, layer_idx, position_idx, k=3, use_transformed=True):
    """
    True RVQ decomposition: iteratively select tokens that best reduce the current residual.
    
    Args:
        use_transformed: If True, use first-layer transformed embeddings. If False, use raw embeddings.
    """
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual
        if layer_idx == -1:
            residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        original_residual = residual.clone()
        
        # Get candidate tokens (top 100 from logit lens for efficiency)
        residual_normed = model.transformer.ln_f(original_residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        vocab_size = model.config.vocab_size
        topk_values, candidate_indices = torch.topk(logits, k=min(20, vocab_size))
        candidate_indices = candidate_indices.tolist()
        
        # Prepare embeddings
        if use_transformed:
            # Pre-compute transformed embeddings for candidates
            candidate_embeddings = {}
            for token_id in candidate_indices:
                try:
                    transformed = get_token_through_first_layer(
                        model, token_id, position_idx, seq_length, device
                    )
                    candidate_embeddings[token_id] = transformed
                except:
                    continue
        else:
            # Use raw embeddings
            all_raw_embeddings = model.transformer.wte.weight
            candidate_embeddings = {tid: all_raw_embeddings[tid] for tid in candidate_indices}
        
        # RVQ: Iteratively select tokens that best reduce current residual
        selected_tokens = []
        selected_coefficients = []
        current_residual = original_residual.clone()
        
        for iteration in range(k):
            best_token = None
            best_coeff = 0
            best_new_residual = None
            best_error = float('inf')
            
            # Find the token that best reduces current residual
            for token_id, embedding in candidate_embeddings.items():
                if token_id in selected_tokens:
                    continue
                
                # Find optimal coefficient for this token
                dot_product = torch.dot(current_residual, embedding)
                embedding_norm_sq = torch.dot(embedding, embedding)
                
                if embedding_norm_sq > 1e-8:  # Avoid division by zero
                    optimal_coeff = dot_product / embedding_norm_sq
                    
                    # Compute new residual after subtracting this token contribution
                    new_residual = current_residual - optimal_coeff * embedding
                    error = torch.norm(new_residual).item()
                    
                    if error < best_error:
                        best_error = error
                        best_token = token_id
                        best_coeff = optimal_coeff.item()
                        best_new_residual = new_residual
            
            if best_token is not None:
                selected_tokens.append(best_token)
                selected_coefficients.append(best_coeff)
                current_residual = best_new_residual.clone()
            else:
                # Fallback: add next available candidate with zero coefficient
                for token_id in candidate_indices:
                    if token_id not in selected_tokens:
                        selected_tokens.append(token_id)
                        selected_coefficients.append(0.0)
                        break
                break
        
        # Compute final reconstruction
        if selected_tokens:
            selected_embeddings = [candidate_embeddings[tid] for tid in selected_tokens if tid in candidate_embeddings]
            if selected_embeddings:
                reconstruction = torch.zeros_like(original_residual)
                for i, (coeff, embedding) in enumerate(zip(selected_coefficients, selected_embeddings)):
                    reconstruction += coeff * embedding
                
                final_error = torch.norm(original_residual - reconstruction).item()
            else:
                reconstruction = torch.zeros_like(original_residual)
                final_error = torch.norm(original_residual).item()
                selected_coefficients = [0.0] * len(selected_tokens)
        else:
            reconstruction = torch.zeros_like(original_residual)
            final_error = torch.norm(original_residual).item()
            selected_coefficients = []
        
        return {
            'residual': original_residual,
            'selected_tokens': [tokenizer.decode([tid]) for tid in selected_tokens],
            'selected_indices': torch.tensor(selected_tokens) if selected_tokens else torch.tensor([]),
            'coefficients': torch.tensor(selected_coefficients) if selected_coefficients else torch.tensor([]),
            'reconstruction': reconstruction,
            'error': final_error,
            'use_transformed': use_transformed
        }


def compare_rvq_vs_simple(model, text, device='cuda'):
    """
    Compare true RVQ method vs simple logit lens (both using first-layer transformation).
    RVQ: Iteratively select tokens that best reduce current residual
    Simple: Select top-k tokens via logit lens, then reconstruct
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    print("COMPARING TRUE RVQ vs SIMPLE LOGIT LENS")
    print("RVQ: Iterative residual-based token selection")
    print("Simple: Direct top-k logit lens selection") 
    print("(Both use first-layer transformation)")
    print("=" * 80)
    
    # Test across different layers
    for layer_idx in [0, 1, 3, 6, 9, 11]:
        print(f"\nLAYER {layer_idx}")
        print("-" * 60)
        
        rvq_errors = []
        simple_errors = []
        improvements = []
        
        for pos_idx in range(min(4, len(tokens))):  # Test first 4 positions
            # True RVQ method (with first-layer transformation)
            rvq_results = true_rvq_decomposition(
                model, input_ids, layer_idx, pos_idx, k=3, use_transformed=True
            )
            
            # Simple logit lens method (with first-layer transformation) 
            simple_results = simple_logit_lens_decomposition(
                model, input_ids, layer_idx, pos_idx, k=3
            )
            
            residual_norm = torch.norm(rvq_results['residual']).item()
            
            # Calculate percentage errors
            rvq_error_pct = 100 * rvq_results['error'] / residual_norm if residual_norm > 0 else 0
            simple_error_pct = 100 * simple_results['error'] / residual_norm if residual_norm > 0 else 0
            improvement = simple_error_pct - rvq_error_pct
            
            rvq_errors.append(rvq_error_pct)
            simple_errors.append(simple_error_pct)
            improvements.append(improvement)
            
            print(f"\nPosition {pos_idx} ('{tokens[pos_idx]}'):")
            print(f"  Simple tokens: {simple_results['topk_tokens']}")
            print(f"  RVQ tokens: {rvq_results['selected_tokens']}")
            print(f"  Residual norm: {residual_norm:.3f}")
            
            print(f"\n  Simple Logit Lens:")
            if len(simple_results['coefficients']) > 0:
                print(f"    Coefficients: {simple_results['coefficients'].cpu().numpy()}")
            print(f"    Error: {simple_results['error']:.3f} ({simple_error_pct:.1f}%)")
            
            print(f"\n  True RVQ (iterative):")
            if len(rvq_results['coefficients']) > 0:
                print(f"    Coefficients: {rvq_results['coefficients'].cpu().numpy()}")
            print(f"    Error: {rvq_results['error']:.3f} ({rvq_error_pct:.1f}%)")
            
            print(f"\n  RVQ advantage: {improvement:.1f} percentage points")
            
            # Show if different tokens were selected
            simple_set = set(simple_results['topk_tokens'])
            rvq_set = set(rvq_results['selected_tokens'])
            if simple_set != rvq_set:
                print(f"  → Different token selections!")
                only_simple = simple_set - rvq_set
                only_rvq = rvq_set - simple_set
                if only_simple:
                    print(f"    Only in Simple: {list(only_simple)}")
                if only_rvq:
                    print(f"    Only in RVQ: {list(only_rvq)}")
        
        # Layer summary
        if rvq_errors and simple_errors:
            avg_simple_error = np.mean(simple_errors)
            avg_rvq_error = np.mean(rvq_errors)
            avg_improvement = np.mean(improvements)
            
            print(f"\nLayer {layer_idx} Summary:")
            print(f"  Simple Logit Lens avg error: {avg_simple_error:.1f}%")
            print(f"  True RVQ avg error: {avg_rvq_error:.1f}%")
            print(f"  RVQ advantage: {avg_improvement:.1f} percentage points")
            
            if avg_improvement > 5:
                print("  → RVQ provides significant improvement")
            elif avg_improvement > 0:
                print("  → RVQ provides modest improvement")
            else:
                print("  → Simple method performs as well or better")


def analyze_first_layer_decomposition(model, text, device='cuda'):
    """
    Analyze decomposition using first-layer transformation across layers.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    
    # Test across different layers
    for layer_idx in [0, 1, 3, 6, 9, 11]:
        print(f"\nLAYER {layer_idx}")
        print("-" * 60)
        
        total_improvement = 0
        count = 0
        
        for pos_idx in range(len(tokens)):
            results = decompose_with_first_layer(
                model, input_ids, layer_idx, pos_idx, k=3
            )
            
            residual_norm = torch.norm(results['residual']).item()
            error_raw_pct = 100 * results['error_raw'] / residual_norm
            error_trans_pct = 100 * results['error_transformed'] / residual_norm
            improvement = error_raw_pct - error_trans_pct
            total_improvement += improvement
            count += 1
            
            print(f"\nPosition {pos_idx} ('{tokens[pos_idx]}'):")
            print(f"  Top tokens: {results['topk_tokens']}")
            print(f"  Residual norm: {residual_norm:.3f}")
            
            print(f"\n  Raw embeddings:")
            print(f"    Coefficients: {results['coeffs_raw'].cpu().numpy()}")
            print(f"    Error: {results['error_raw']:.3f} ({error_raw_pct:.1f}%)")
            print(f"    Cosine sims: {[f'{results[f'cosine_raw_{i}']:.3f}' for i in range(3)]}")
            
            print(f"\n  First-layer transformed:")
            print(f"    Coefficients: {results['coeffs_transformed'].cpu().numpy()}")
            print(f"    Error: {results['error_transformed']:.3f} ({error_trans_pct:.1f}%)")
            print(f"    Cosine sims: {[f'{results[f'cosine_transformed_{i}']:.3f}' for i in range(3)]}")
            print(f"\n  Improvement: {improvement:.1f} percentage points")
            
            # Show norms to understand scale differences
            raw_norms = [torch.norm(results['raw_embeddings'][i]).item() for i in range(3)]
            trans_norms = [torch.norm(results['transformed_embeddings'][i]).item() for i in range(3)]
            print(f"\n  Embedding norms:")
            print(f"    Raw: {raw_norms}")
            print(f"    Transformed: {trans_norms}")
        
        avg_improvement = total_improvement / count
        print(f"\nLayer {layer_idx} average improvement: {avg_improvement:.1f} percentage points")


def compare_approaches_visual(model, text, device='cuda'):
    """
    Visualize the comparison between raw and transformed embeddings.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    n_layers = 12
    n_positions = len(tokens)
    
    errors_raw = np.zeros((n_layers, n_positions))
    errors_transformed = np.zeros((n_layers, n_positions))
    
    for layer_idx in range(n_layers):
        for pos_idx in range(n_positions):
            results = decompose_with_first_layer(
                model, input_ids, layer_idx, pos_idx, k=3
            )
            
            residual_norm = torch.norm(results['residual']).item()
            errors_raw[layer_idx, pos_idx] = 100 * results['error_raw'] / residual_norm
            errors_transformed[layer_idx, pos_idx] = 100 * results['error_transformed'] / residual_norm
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Raw embedding errors
    im1 = axes[0, 0].imshow(errors_raw, aspect='auto', cmap='viridis', vmin=0, vmax=100)
    axes[0, 0].set_title('Raw Embedding Errors (%)')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Layer')
    axes[0, 0].set_xticks(range(n_positions))
    axes[0, 0].set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Transformed embedding errors
    im2 = axes[0, 1].imshow(errors_transformed, aspect='auto', cmap='viridis', vmin=0, vmax=100)
    axes[0, 1].set_title('First-Layer Transformed Errors (%)')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Layer')
    axes[0, 1].set_xticks(range(n_positions))
    axes[0, 1].set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Improvement
    improvement = errors_raw - errors_transformed
    im3 = axes[1, 0].imshow(improvement, aspect='auto', cmap='RdBu_r', 
                            vmin=-20, vmax=20)
    axes[1, 0].set_title('Improvement (Raw - Transformed)')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Layer')
    axes[1, 0].set_xticks(range(n_positions))
    axes[1, 0].set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Average by layer
    axes[1, 1].plot(range(n_layers), errors_raw.mean(axis=1), 
                   label='Raw', marker='o')
    axes[1, 1].plot(range(n_layers), errors_transformed.mean(axis=1), 
                   label='Transformed', marker='s')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Average Error (%)')
    axes[1, 1].set_title('Average Error by Layer')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return errors_raw, errors_transformed


def test_different_contexts(model, token_str, device='cuda'):
    """
    Test how the same token transforms differently with different contexts.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]
    
    print(f"\nTesting token '{token_str}' (id={token_id}) in different contexts")
    print("=" * 80)
    
    # Test in different positions and sequence lengths
    positions = [0, 2, 5, 9]
    seq_lengths = [10, 10, 10, 10]
    
    transformations = []
    
    for pos, seq_len in zip(positions, seq_lengths):
        if pos >= seq_len:
            continue
            
        transformed = get_token_through_first_layer(
            model, token_id, pos, seq_len, device
        )
        
        norm = torch.norm(transformed).item()
        transformations.append(transformed)
        
        print(f"\nPosition {pos} (seq_length={seq_len}):")
        print(f"  Norm: {norm:.3f}")
        print(f"  Mean: {transformed.mean().item():.6f}")
        print(f"  Std: {transformed.std().item():.6f}")
    
    # Compare transformations
    print("\nCosine similarities between different positions:")
    for i in range(len(transformations)):
        for j in range(i+1, len(transformations)):
            cos_sim = F.cosine_similarity(
                transformations[i].unsqueeze(0),
                transformations[j].unsqueeze(0)
            ).item()
            print(f"  Position {positions[i]} vs {positions[j]}: {cos_sim:.3f}")
    
    return transformations


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The girl was named"
    
    print("=" * 80)
    print("RVQ vs SIMPLE LOGIT LENS COMPARISON")
    print("=" * 80)
    
    # Direct comparison between methods
    compare_rvq_vs_simple(model, text, device)
    
    print("\n" + "=" * 80)
    print("FIRST LAYER TRANSFORMATION EXPERIMENT (DETAILED)")
    print("=" * 80)
    
    # Main analysis
    analyze_first_layer_decomposition(model, text, device)
    
    # Visual comparison
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    errors_raw, errors_transformed = compare_approaches_visual(model, text, device)
    
    # Test context dependency
    print("\n" + "=" * 80)
    print("CONTEXT DEPENDENCY TEST")
    test_different_contexts(model, " fox", device)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    avg_raw = errors_raw.mean()
    avg_transformed = errors_transformed.mean()
    improvement = avg_raw - avg_transformed
    
    print(f"Overall average error:")
    print(f"  Raw embeddings: {avg_raw:.1f}%")
    print(f"  First-layer transformed: {avg_transformed:.1f}%")
    print(f"  Improvement: {improvement:.1f} percentage points")
    
    if improvement > 5:
        print("\n✓ First layer transformation significantly improves reconstruction!")
    elif improvement > 0:
        print("\n→ First layer transformation provides modest improvement")
    else:
        print("\n✗ First layer transformation doesn't help (or makes it worse)")


if __name__ == "__main__":
    main()