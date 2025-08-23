import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def get_residual_and_decompose_rvq(model, input_ids, layer_idx, position_idx, 
                                   n_iterations=3, tokens_per_iter=1):
    """
    RVQ version: Iteratively decompose residual by subtracting best approximation
    at each step and finding tokens that best explain the remainder.
    
    Args:
        n_iterations: Number of RVQ iterations
        tokens_per_iter: Tokens to select per iteration
    """
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get initial residual
        if layer_idx == -1:
            original_residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            original_residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Store results for each iteration
        rvq_components = []
        current_residual = original_residual.clone()
        
        for iteration in range(n_iterations):
            # Apply LogitLens to current residual
            residual_normed = model.transformer.ln_f(current_residual.unsqueeze(0))
            logits = model.lm_head(residual_normed)
            
            # Get top token(s) for this iteration
            top_values, top_indices = torch.topk(logits[0], k=tokens_per_iter)
            top_embeddings = model.transformer.wte(top_indices)
            
            # Find best coefficient to minimize residual
            # Solve: min ||current_residual - alpha * embedding||
            if tokens_per_iter == 1:
                embedding = top_embeddings[0]
                # Optimal alpha = (residual · embedding) / (embedding · embedding)
                alpha = torch.dot(current_residual, embedding) / torch.dot(embedding, embedding)
                approximation = alpha * embedding
                
                component_info = {
                    'iteration': iteration,
                    'token': tokenizer.decode([top_indices[0].item()]),
                    'token_id': top_indices[0].item(),
                    'logit': top_values[0].item(),
                    'coefficient': alpha.item(),
                    'embedding': embedding,
                    'approximation': approximation,
                    'residual_before': current_residual.clone(),
                }
            else:
                # Multiple tokens - use least squares
                A = top_embeddings.T
                b = current_residual.unsqueeze(1)
                alphas = torch.linalg.lstsq(A, b).solution.squeeze()
                approximation = (A @ alphas.unsqueeze(1)).squeeze()
                
                component_info = {
                    'iteration': iteration,
                    'tokens': [tokenizer.decode([idx.item()]) for idx in top_indices],
                    'token_ids': top_indices.tolist(),
                    'logits': top_values.tolist(),
                    'coefficients': alphas.cpu().numpy(),
                    'embeddings': top_embeddings,
                    'approximation': approximation,
                    'residual_before': current_residual.clone(),
                }
            
            # Compute similarity before subtraction
            cos_sim = F.cosine_similarity(
                current_residual.unsqueeze(0),
                approximation.unsqueeze(0)
            ).item()
            component_info['cosine_similarity'] = cos_sim
            
            # Update residual for next iteration
            current_residual = current_residual - approximation
            component_info['residual_after'] = current_residual.clone()
            component_info['residual_norm_before'] = torch.norm(component_info['residual_before']).item()
            component_info['residual_norm_after'] = torch.norm(current_residual).item()
            component_info['approximation_norm'] = torch.norm(approximation).item()
            
            rvq_components.append(component_info)
        
        # Final decomposition
        decomposed = {
            'original': original_residual,
            'components': rvq_components,
            'final_residual': current_residual,
            'reconstruction': original_residual - current_residual
        }
        
        return decomposed


def analyze_rvq_decomposition(model, text, layer_idx=5, n_iterations=5, device='cuda'):
    """
    Analyze RVQ decomposition at a specific layer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Analyzing layer {layer_idx}")
    print("=" * 80)
    
    results_by_position = []
    
    for pos_idx in range(len(tokens)):
        print(f"\nPosition {pos_idx} ('{tokens[pos_idx]}'):")
        print("-" * 60)
        
        decomposed = get_residual_and_decompose_rvq(
            model, input_ids, layer_idx, pos_idx, 
            n_iterations=n_iterations, tokens_per_iter=1
        )
        
        original_norm = torch.norm(decomposed['original']).item()
        
        print(f"Original norm: {original_norm:.3f}")
        print("\nRVQ Iterations:")
        
        cumulative_reduction = 0
        for comp in decomposed['components']:
            iter_num = comp['iteration']
            token = comp['token']
            coeff = comp['coefficient']
            cos_sim = comp['cosine_similarity']
            norm_before = comp['residual_norm_before']
            norm_after = comp['residual_norm_after']
            reduction = (norm_before - norm_after) / original_norm * 100
            cumulative_reduction += reduction
            
            print(f"  Iter {iter_num}: '{token:15s}' "
                  f"coeff={coeff:7.3f} "
                  f"cos_sim={cos_sim:6.3f} "
                  f"reduction={reduction:5.1f}% "
                  f"cumulative={cumulative_reduction:5.1f}%")
        
        final_error = torch.norm(decomposed['final_residual']).item()
        final_error_pct = 100 * final_error / original_norm
        print(f"\nFinal residual: {final_error:.3f} ({final_error_pct:.1f}% of original)")
        
        results_by_position.append(decomposed)
    
    return results_by_position


def compare_rvq_vs_simultaneous(model, text, layer_idx=5, device='cuda'):
    """
    Compare RVQ (sequential) vs simultaneous decomposition.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print("\n" + "=" * 80)
    print("COMPARING RVQ vs SIMULTANEOUS DECOMPOSITION")
    print("=" * 80)
    
    pos_idx = min(2, len(input_ids[0]) - 1)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        original = outputs.hidden_states[layer_idx][0, pos_idx, :]
        original_norm = torch.norm(original).item()
        
        # Method 1: Simultaneous (all at once)
        residual_normed = model.transformer.ln_f(original.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        top3_indices = torch.topk(logits[0], k=3).indices
        top3_embeddings = model.transformer.wte(top3_indices)
        
        # Least squares for simultaneous
        A = top3_embeddings.T
        b = original.unsqueeze(1)
        coeffs_simul = torch.linalg.lstsq(A, b).solution.squeeze()
        recon_simul = (A @ coeffs_simul.unsqueeze(1)).squeeze()
        error_simul = torch.norm(original - recon_simul).item()
        
        tokens_simul = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    # Method 2: RVQ (sequential)
    decomposed_rvq = get_residual_and_decompose_rvq(
        model, input_ids, layer_idx, pos_idx, 
        n_iterations=3, tokens_per_iter=1
    )
    
    error_rvq = torch.norm(decomposed_rvq['final_residual']).item()
    
    print(f"Position {pos_idx}, Layer {layer_idx}")
    print(f"Original norm: {original_norm:.3f}")
    
    print("\n1. SIMULTANEOUS (all 3 tokens at once):")
    print(f"   Tokens: {tokens_simul}")
    print(f"   Coefficients: {coeffs_simul.cpu().numpy()}")
    print(f"   Error: {error_simul:.3f} ({100*error_simul/original_norm:.1f}%)")
    
    print("\n2. RVQ (3 iterations, 1 token each):")
    for comp in decomposed_rvq['components']:
        print(f"   Iter {comp['iteration']}: '{comp['token']:15s}' coeff={comp['coefficient']:7.3f}")
    print(f"   Error: {error_rvq:.3f} ({100*error_rvq/original_norm:.1f}%)")
    
    print(f"\nRVQ vs Simultaneous error ratio: {error_rvq/error_simul:.3f}")


def visualize_rvq_progression(model, text, layer_idx=5, device='cuda'):
    """
    Visualize how RVQ progressively reduces the residual.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    n_positions = len(tokens)
    n_iterations = 10
    
    # Collect data
    reduction_curves = []
    
    for pos_idx in range(n_positions):
        decomposed = get_residual_and_decompose_rvq(
            model, input_ids, layer_idx, pos_idx,
            n_iterations=n_iterations, tokens_per_iter=1
        )
        
        original_norm = torch.norm(decomposed['original']).item()
        norms = [original_norm]
        
        for comp in decomposed['components']:
            norms.append(comp['residual_norm_after'])
        
        # Convert to percentage of original
        reduction_pcts = [100 * (1 - n/original_norm) for n in norms]
        reduction_curves.append(reduction_pcts)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Reduction curves
    plt.subplot(1, 2, 1)
    for pos_idx, curve in enumerate(reduction_curves):
        plt.plot(range(len(curve)), curve, marker='o', label=f"Pos {pos_idx}: '{tokens[pos_idx]}'")
    
    plt.xlabel('RVQ Iteration')
    plt.ylabel('Reduction in Residual Norm (%)')
    plt.title(f'RVQ Decomposition Progress (Layer {layer_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final errors
    plt.subplot(1, 2, 2)
    final_reductions = [curve[-1] for curve in reduction_curves]
    positions = range(n_positions)
    plt.bar(positions, final_reductions)
    plt.xlabel('Position')
    plt.ylabel('Final Reduction (%)')
    plt.title(f'Final Reduction by Position (after {n_iterations} iterations)')
    plt.xticks(positions, tokens, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return reduction_curves


def analyze_rvq_token_patterns(model, text, device='cuda'):
    """
    Analyze which tokens appear most frequently in RVQ decompositions.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print("\n" + "=" * 80)
    print("RVQ TOKEN PATTERNS ACROSS LAYERS")
    print("=" * 80)
    
    # Collect token frequencies
    from collections import defaultdict
    token_counts = defaultdict(lambda: defaultdict(int))
    
    for layer_idx in [0, 3, 6, 9, 11]:
        for pos_idx in range(len(input_ids[0])):
            decomposed = get_residual_and_decompose_rvq(
                model, input_ids, layer_idx, pos_idx,
                n_iterations=5, tokens_per_iter=1
            )
            
            for i, comp in enumerate(decomposed['components']):
                token = comp['token']
                token_counts[layer_idx][token] += 1
    
    # Display top tokens per layer
    for layer_idx in sorted(token_counts.keys()):
        print(f"\nLayer {layer_idx} - Most frequent RVQ tokens:")
        sorted_tokens = sorted(token_counts[layer_idx].items(), 
                              key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens[:5]:
            print(f"  '{token:15s}': {count} times")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox jumps over the lazy dog"
    
    # Basic RVQ analysis
    print("=" * 80)
    print("RVQ DECOMPOSITION ANALYSIS")
    print("=" * 80)
    results = analyze_rvq_decomposition(model, text, layer_idx=5, n_iterations=5, device=device)
    
    # Compare methods
    compare_rvq_vs_simultaneous(model, text, layer_idx=5, device=device)
    
    # Visualize progression
    print("\nVisualizing RVQ progression...")
    reduction_curves = visualize_rvq_progression(model, text, layer_idx=5, device=device)
    
    # Analyze patterns
    analyze_rvq_token_patterns(model, text, device=device)
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. RVQ iteratively removes the 'loudest' component at each step")
    print("2. Later iterations find tokens that explain what's left")
    print("3. RVQ often achieves similar or better reconstruction than simultaneous")
    print("4. The sequence of tokens reveals hierarchical structure in the representation")
    print("5. Different from simultaneous: tokens chosen depend on previous choices")


if __name__ == "__main__":
    main()