import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

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


def decompose_residual(model, input_ids, layer_idx, position_idx, k=3):
    """Decompose residual using both raw and first-layer transformed embeddings."""
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual
        residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via LogitLens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        topk_values, topk_indices = torch.topk(logits, k=k)
        
        # Method 1: Raw embeddings
        raw_embeddings = model.transformer.wte(topk_indices)
        
        # Method 2: First-layer transformed tokens
        # For layer 0, use raw embeddings for both methods
        if layer_idx == 0:
            transformed_embeddings = raw_embeddings
        else:
            transformed_embeddings = []
            for token_id in topk_indices:
                transformed = get_token_through_first_layer(
                    model, token_id, position_idx, seq_length, device
                )
                transformed_embeddings.append(transformed)
            transformed_embeddings = torch.stack(transformed_embeddings)
        
        # Reconstruct using least squares for both methods
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
        
        return {
            'error_raw': error_raw,
            'error_transformed': error_trans,
            'residual_norm': torch.norm(residual).item()
        }


def run_single_prompt_experiment(model, text, device='cuda', k=3):
    """Run experiment on a single prompt across all layers."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    layers = list(range(12))
    n_positions = len(tokens)
    
    raw_errors = []
    transformed_errors = []
    
    for layer_idx in layers:
        layer_raw_errors = []
        layer_trans_errors = []
        
        for pos_idx in range(n_positions):
            result = decompose_residual(model, input_ids, layer_idx, pos_idx, k=k)
            
            # Convert to percentage error
            residual_norm = result['residual_norm']
            raw_error_pct = 100 * result['error_raw'] / residual_norm if residual_norm > 0 else 0
            trans_error_pct = 100 * result['error_transformed'] / residual_norm if residual_norm > 0 else 0
            
            layer_raw_errors.append(raw_error_pct)
            layer_trans_errors.append(trans_error_pct)
        
        # Average across positions for this layer
        avg_raw_error = np.mean(layer_raw_errors)
        avg_trans_error = np.mean(layer_trans_errors)
        
        raw_errors.append(avg_raw_error)
        transformed_errors.append(avg_trans_error)
    
    return raw_errors, transformed_errors


def run_multi_prompt_experiment(model, prompts, device='cuda', k=3):
    """Run experiment across multiple prompts to get confidence intervals."""
    layers = list(range(12))
    all_raw_errors = []
    all_transformed_errors = []
    
    print(f"Running experiment on {len(prompts)} prompts:")
    for i, text in enumerate(prompts):
        print(f"  {i+1:2d}. '{text}'")
    print("=" * 70)
    
    for i, text in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: '{text}'")
        
        raw_errors, transformed_errors = run_single_prompt_experiment(
            model, text, device, k
        )
        
        all_raw_errors.append(raw_errors)
        all_transformed_errors.append(transformed_errors)
    
    # Convert to numpy arrays for easier statistics
    all_raw_errors = np.array(all_raw_errors)  # shape: (n_prompts, n_layers)
    all_transformed_errors = np.array(all_transformed_errors)
    
    # Calculate statistics across prompts
    raw_means = np.mean(all_raw_errors, axis=0)
    raw_stds = np.std(all_raw_errors, axis=0)
    raw_sems = raw_stds / np.sqrt(len(prompts))  # Standard error of mean
    
    transformed_means = np.mean(all_transformed_errors, axis=0)
    transformed_stds = np.std(all_transformed_errors, axis=0)
    transformed_sems = transformed_stds / np.sqrt(len(prompts))
    
    return {
        'layers': layers,
        'raw_means': raw_means,
        'raw_stds': raw_stds,
        'raw_sems': raw_sems,
        'transformed_means': transformed_means,
        'transformed_stds': transformed_stds,
        'transformed_sems': transformed_sems,
        'all_raw_errors': all_raw_errors,
        'all_transformed_errors': all_transformed_errors,
        'prompts': prompts
    }


def create_confidence_plot(results, save_path='poster_figure_with_confidence.png'):
    """Create a plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    layers = results['layers']
    
    # Plot means
    line1 = ax.plot(layers, results['raw_means'], 'o-', linewidth=2.5, markersize=8, 
                    label='Raw Token Embeddings', color='#e74c3c', zorder=3)
    line2 = ax.plot(layers, results['transformed_means'], 's-', linewidth=2.5, markersize=8, 
                    label='First-Layer Transformed', color='#3498db', zorder=3)
    
    # Add confidence intervals (±1 SEM, roughly 68% confidence)
    ax.fill_between(layers, 
                    results['raw_means'] - results['raw_sems'],
                    results['raw_means'] + results['raw_sems'],
                    color='#e74c3c', alpha=0.2, zorder=1)
    
    ax.fill_between(layers,
                    results['transformed_means'] - results['transformed_sems'],
                    results['transformed_means'] + results['transformed_sems'],
                    color='#3498db', alpha=0.2, zorder=1)
    
    # Formatting
    ax.set_xlabel('Layer', fontsize=16)
    ax.set_ylabel('Reconstruction Error (%)', fontsize=16)
    ax.set_title(f'Token Embedding Reconstruction Error by Layer\n'
                f'Averaged over {len(results["prompts"])} prompts (±SEM)', 
                fontsize=18, pad=20)
    
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xticks(layers)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add improvement annotation (excluding layer 0)
    improvement = np.mean(results['raw_means'][1:]) - np.mean(results['transformed_means'][1:])
    ax.text(0.02, 0.98, f'Average improvement (layers 1-11): {improvement:.1f} percentage points\n'
                        f'n = {len(results["prompts"])} prompts', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def print_detailed_results(results):
    """Print detailed statistics."""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS FOR POSTER")
    print("=" * 70)
    
    print(f"Number of prompts: {len(results['prompts'])}")
    print(f"Prompts tested:")
    for i, prompt in enumerate(results['prompts']):
        print(f"  {i+1:2d}. '{prompt}'")
    
    print(f"\nLayer-by-layer results (mean ± SEM):")
    print(f"{'Layer':<5} {'Raw Error (%)':<15} {'Trans Error (%)':<15} {'Improvement':<12}")
    print("-" * 70)
    
    for i, layer in enumerate(results['layers']):
        raw_mean = results['raw_means'][i]
        raw_sem = results['raw_sems'][i]
        trans_mean = results['transformed_means'][i]
        trans_sem = results['transformed_sems'][i]
        improvement = raw_mean - trans_mean
        
        if layer == 0:
            print(f"{layer:<5} {raw_mean:>6.1f} ± {raw_sem:>4.1f}   {trans_mean:>6.1f} ± {trans_sem:>4.1f}   "
                  f"{'(no transform)':>12}")
        else:
            print(f"{layer:<5} {raw_mean:>6.1f} ± {raw_sem:>4.1f}   {trans_mean:>6.1f} ± {trans_sem:>4.1f}   "
                  f"{improvement:>+6.1f}pp")
    
    # Overall statistics (excluding layer 0)
    print(f"\nOverall statistics (layers 1-11):")
    avg_raw = np.mean(results['raw_means'][1:])
    avg_trans = np.mean(results['transformed_means'][1:])
    overall_improvement = avg_raw - avg_trans
    
    print(f"  Raw embeddings:           {avg_raw:.1f}%")
    print(f"  First-layer transformed:  {avg_trans:.1f}%")
    print(f"  Overall improvement:      {overall_improvement:.1f} percentage points")
    
    # Statistical significance (rough check)
    improvements = results['all_raw_errors'][:, 1:] - results['all_transformed_errors'][:, 1:]
    mean_improvement_per_prompt = np.mean(improvements, axis=1)
    overall_sem = np.std(mean_improvement_per_prompt) / np.sqrt(len(results['prompts']))
    
    print(f"\nStatistical notes:")
    print(f"  Mean improvement per prompt: {np.mean(mean_improvement_per_prompt):.1f} ± {overall_sem:.1f}pp")
    print(f"  All prompts show improvement: {np.all(mean_improvement_per_prompt > 0)}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Multiple prompts for confidence intervals
    prompts = [
        "The cat sat",
        "She was walking",
        "John likes pizza",
        "The book contains",
        "We are going",
        "It was raining",
        "The dog ran",
        "They were singing"
    ]
    
    k = 3  # Number of top tokens to use
    
    # Run multi-prompt experiment
    results = run_multi_prompt_experiment(model, prompts, device, k)
    
    # Create plot with confidence intervals
    save_path = create_confidence_plot(results)
    print(f"\nPoster figure with confidence intervals saved as: {save_path}")
    
    # Print detailed results
    print_detailed_results(results)


if __name__ == "__main__":
    main()