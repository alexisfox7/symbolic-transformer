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


def decompose_residual_fast(model, input_ids, layer_idx, k=3):
    """Fast version: only use first position to speed up."""
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    position_idx = 0  # Only use first position for speed
    
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
    """Run experiment on a single prompt across layers 1-11 only."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Only layers 1-11 (exclude layer 0)
    layers = list(range(1, 12))
    
    raw_errors = []
    transformed_errors = []
    
    for layer_idx in layers:
        result = decompose_residual_fast(model, input_ids, layer_idx, k)
        
        # Convert to percentage error
        residual_norm = result['residual_norm']
        raw_error_pct = 100 * result['error_raw'] / residual_norm if residual_norm > 0 else 0
        trans_error_pct = 100 * result['error_transformed'] / residual_norm if residual_norm > 0 else 0
        
        raw_errors.append(raw_error_pct)
        transformed_errors.append(trans_error_pct)
    
    return raw_errors, transformed_errors


def run_multi_prompt_experiment(model, prompts, device='cuda', k=3):
    """Run experiment across multiple prompts to get confidence intervals."""
    layers = list(range(1, 12))  # Only layers 1-11
    all_raw_errors = []
    all_transformed_errors = []
    
    print(f"Running experiment on {len(prompts)} prompts (layers 1-11 only):")
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


def create_final_poster_plot(results, save_path='final_poster_figure.png'):
    """Create the final poster plot excluding layer 0."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    layers = results['layers']
    
    # Plot means
    line1 = ax.plot(layers, results['raw_means'], 'o-', linewidth=2.5, markersize=8, 
                    label='Raw Token Embeddings', color='#e74c3c', zorder=3)
    line2 = ax.plot(layers, results['transformed_means'], 's-', linewidth=2.5, markersize=8, 
                    label='First-Layer Transformed', color='#3498db', zorder=3)
    
    # Add confidence intervals (±1 SEM)
    ax.fill_between(layers, 
                    results['raw_means'] - results['raw_sems'],
                    results['raw_means'] + results['raw_sems'],
                    color='#e74c3c', alpha=0.25, zorder=1, label='_nolegend_')
    
    ax.fill_between(layers,
                    results['transformed_means'] - results['transformed_sems'],
                    results['transformed_means'] + results['transformed_sems'],
                    color='#3498db', alpha=0.25, zorder=1, label='_nolegend_')
    
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
    
    # Add improvement annotation
    improvement = np.mean(results['raw_means']) - np.mean(results['transformed_means'])
    ax.text(0.02, 0.98, f'Average improvement (layers 1-11): {improvement:.1f} percentage points\n'
                        f'n = {len(results["prompts"])} prompts', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def print_final_results(results):
    """Print final results excluding layer 0."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS FOR POSTER (LAYERS 1-11 ONLY)")
    print("=" * 70)
    
    print(f"Number of prompts: {len(results['prompts'])}")
    print(f"Layers analyzed: 1-11 (layer 0 excluded)")
    print(f"Analysis: First position only")
    
    # Overall statistics
    avg_raw = np.mean(results['raw_means'])
    avg_trans = np.mean(results['transformed_means'])
    overall_improvement = avg_raw - avg_trans
    
    print(f"\nOverall statistics (layers 1-11):")
    print(f"  Raw embeddings:           {avg_raw:.1f}% ± {np.mean(results['raw_sems']):.1f}")
    print(f"  First-layer transformed:  {avg_trans:.1f}% ± {np.mean(results['transformed_sems']):.1f}")
    print(f"  Overall improvement:      {overall_improvement:.1f} percentage points")
    
    # Layer-by-layer breakdown
    print(f"\nLayer-by-layer results:")
    print(f"{'Layer':<5} {'Raw (%)':<10} {'Trans (%)':<10} {'Improvement':<12}")
    print("-" * 50)
    
    for i, layer in enumerate(results['layers']):
        raw_mean = results['raw_means'][i]
        trans_mean = results['transformed_means'][i]
        improvement = raw_mean - trans_mean
        
        print(f"{layer:<5} {raw_mean:>6.1f}     {trans_mean:>6.1f}     {improvement:>+6.1f}pp")
    
    # Statistical significance check
    improvements = results['all_raw_errors'] - results['all_transformed_errors']
    mean_improvement_per_prompt = np.mean(improvements, axis=1)
    overall_sem = np.std(mean_improvement_per_prompt) / np.sqrt(len(results['prompts']))
    
    print(f"\nStatistical notes:")
    print(f"  Mean improvement per prompt: {np.mean(mean_improvement_per_prompt):.1f} ± {overall_sem:.1f}pp")
    print(f"  All prompts show improvement: {np.all(mean_improvement_per_prompt > 0)}")
    
    # Show range of improvements by layer
    layer_improvements = results['raw_means'] - results['transformed_means']
    print(f"  Layer improvement range: {np.min(layer_improvements):.1f}pp to {np.max(layer_improvements):.1f}pp")
    
    # Effect size
    effect_size = overall_improvement / np.mean(results['raw_means'])
    print(f"  Relative improvement: {effect_size*100:.1f}% reduction in error")


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
        "It was raining"
    ]
    
    k = 3  # Number of top tokens to use
    
    # Run multi-prompt experiment (layers 1-11 only)
    results = run_multi_prompt_experiment(model, prompts, device, k)
    
    # Create final poster plot
    save_path = create_final_poster_plot(results)
    print(f"\nFinal poster figure saved as: {save_path}")
    
    # Print final results
    print_final_results(results)


if __name__ == "__main__":
    main()