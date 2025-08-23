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


def compute_token_cosine_similarities(model, input_ids, layer_idx):
    """Compare cosine similarity between residual and top-1 predicted token (raw vs transformed)."""
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    seq_length = input_ids.shape[1]
    position_idx = 0  # Only use first position for speed
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual at this layer
        residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top 1 predicted token via LogitLens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        top1_value, top1_index = torch.topk(logits, k=1)
        
        top1_token_id = top1_index[0].item()
        top1_token_str = tokenizer.decode([top1_token_id])
        
        # Method 1: Raw token embedding
        raw_embedding = model.transformer.wte(top1_index)[0]  # Shape: [d_model]
        
        # Method 2: First-layer transformed token embedding
        transformed_embedding = get_token_through_first_layer(
            model, top1_token_id, position_idx, seq_length, device
        )
        
        # Compute cosine similarities between residual and each embedding
        cosine_sim_raw = F.cosine_similarity(
            residual.unsqueeze(0), 
            raw_embedding.unsqueeze(0)
        ).item()
        
        cosine_sim_transformed = F.cosine_similarity(
            residual.unsqueeze(0), 
            transformed_embedding.unsqueeze(0)
        ).item()
        
        return {
            'cosine_sim_raw': cosine_sim_raw,
            'cosine_sim_transformed': cosine_sim_transformed,
            'top1_token': top1_token_str,
            'top1_token_id': top1_token_id,
            'top1_logit': top1_value.item(),
            'residual_norm': torch.norm(residual).item(),
            'raw_embedding_norm': torch.norm(raw_embedding).item(),
            'transformed_embedding_norm': torch.norm(transformed_embedding).item()
        }


def run_single_prompt_experiment(model, text, device='cuda'):
    """Run experiment on a single prompt across layers 1-11."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Only layers 1-11 (exclude layer 0)
    layers = list(range(1, 12))
    
    raw_cosines = []
    transformed_cosines = []
    top1_tokens = []
    
    for layer_idx in layers:
        result = compute_token_cosine_similarities(model, input_ids, layer_idx)
        
        # Store cosine similarities and token info
        raw_cosines.append(result['cosine_sim_raw'])
        transformed_cosines.append(result['cosine_sim_transformed'])
        top1_tokens.append(result['top1_token'])
    
    return raw_cosines, transformed_cosines, top1_tokens


def run_multi_prompt_experiment(model, prompts, device='cuda'):
    """Run experiment across multiple prompts to get confidence intervals."""
    layers = list(range(1, 12))  # Only layers 1-11
    all_raw_cosines = []
    all_transformed_cosines = []
    all_top1_tokens = []
    
    print(f"Running TOKEN COSINE SIMILARITY experiment on {len(prompts)} prompts:")
    print("Comparing residual vs top-1 predicted token (raw vs first-layer transformed)")
    for i, text in enumerate(prompts):
        print(f"  {i+1:2d}. '{text}'")
    print("=" * 70)
    
    for i, text in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: '{text}'")
        
        raw_cosines, transformed_cosines, top1_tokens = run_single_prompt_experiment(
            model, text, device
        )
        
        all_raw_cosines.append(raw_cosines)
        all_transformed_cosines.append(transformed_cosines)
        all_top1_tokens.append(top1_tokens)
        
        # Show example tokens for first prompt
        if i == 0:
            print(f"  Example top-1 tokens by layer: {top1_tokens[:5]}...")
    
    # Convert to numpy arrays for easier statistics
    all_raw_cosines = np.array(all_raw_cosines)  # shape: (n_prompts, n_layers)
    all_transformed_cosines = np.array(all_transformed_cosines)
    
    # Calculate statistics across prompts
    raw_means = np.mean(all_raw_cosines, axis=0)
    raw_stds = np.std(all_raw_cosines, axis=0)
    raw_sems = raw_stds / np.sqrt(len(prompts))  # Standard error of mean
    
    transformed_means = np.mean(all_transformed_cosines, axis=0)
    transformed_stds = np.std(all_transformed_cosines, axis=0)
    transformed_sems = transformed_stds / np.sqrt(len(prompts))
    
    return {
        'layers': layers,
        'raw_means': raw_means,
        'raw_stds': raw_stds,
        'raw_sems': raw_sems,
        'transformed_means': transformed_means,
        'transformed_stds': transformed_stds,
        'transformed_sems': transformed_sems,
        'all_raw_cosines': all_raw_cosines,
        'all_transformed_cosines': all_transformed_cosines,
        'all_top1_tokens': all_top1_tokens,
        'prompts': prompts
    }


def create_token_cosine_plot(results, save_path='token_cosine_similarity_poster.png'):
    """Create the poster plot for token cosine similarity comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    layers = results['layers']
    
    # Plot means
    line1 = ax.plot(layers, results['raw_means'], 'o-', linewidth=2.5, markersize=8, 
                    label='Raw Token Embedding', color='#e74c3c', zorder=3)
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
    ax.set_ylabel('Cosine Similarity', fontsize=16)
    ax.set_title(f'Residual vs Top-1 Token Cosine Similarity by Layer\n'
                f'Raw vs First-Layer Transformed Tokens (n={len(results["prompts"])} prompts ±SEM)', 
                fontsize=18, pad=20)
    
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xticks(layers)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Set y-axis range to show full cosine similarity range
    ax.set_ylim(-0.1, 1.0)
    
    # Add improvement annotation
    improvement = np.mean(results['transformed_means']) - np.mean(results['raw_means'])
    
    ax.text(0.02, 0.98, f'Average improvement: +{improvement:.3f} cosine similarity\n'
                        f'Residual ↔ Top-1 predicted token alignment', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def print_token_cosine_results(results):
    """Print results for token cosine similarity analysis."""
    print("\n" + "=" * 70)
    print("TOKEN COSINE SIMILARITY RESULTS")
    print("=" * 70)
    
    print(f"Number of prompts: {len(results['prompts'])}")
    print(f"Analysis: Cosine similarity between layer residual and top-1 predicted token")
    print(f"Comparison: Raw token embedding vs First-layer transformed token")
    print(f"Range: -1 (opposite) to +1 (perfect alignment)")
    
    # Overall statistics
    avg_raw = np.mean(results['raw_means'])
    avg_trans = np.mean(results['transformed_means'])
    overall_improvement = avg_trans - avg_raw  # For cosine sim, higher is better
    
    print(f"\nOverall statistics (layers 1-11):")
    print(f"  Raw token embedding:      {avg_raw:.3f} ± {np.mean(results['raw_sems']):.3f}")
    print(f"  First-layer transformed:  {avg_trans:.3f} ± {np.mean(results['transformed_sems']):.3f}")
    print(f"  Overall improvement:      +{overall_improvement:.3f} cosine similarity")
    
    # Layer-by-layer breakdown
    print(f"\nLayer-by-layer results:")
    print(f"{'Layer':<5} {'Raw Token':<12} {'Transformed':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for i, layer in enumerate(results['layers']):
        raw_mean = results['raw_means'][i]
        trans_mean = results['transformed_means'][i]
        improvement = trans_mean - raw_mean
        
        print(f"{layer:<5} {raw_mean:>8.3f}     {trans_mean:>8.3f}       {improvement:>+7.3f}")
    
    # Statistical significance check
    improvements = results['all_transformed_cosines'] - results['all_raw_cosines']
    mean_improvement_per_prompt = np.mean(improvements, axis=1)
    overall_sem = np.std(mean_improvement_per_prompt) / np.sqrt(len(results['prompts']))
    
    print(f"\nStatistical notes:")
    print(f"  Mean improvement per prompt: +{np.mean(mean_improvement_per_prompt):.3f} ± {overall_sem:.3f}")
    print(f"  Prompts showing improvement: {np.sum(mean_improvement_per_prompt > 0)}/{len(results['prompts'])}")
    
    # Show range of improvements by layer
    layer_improvements = results['transformed_means'] - results['raw_means']
    best_layer = np.argmax(layer_improvements) + 1  # +1 because we start from layer 1
    worst_layer = np.argmin(layer_improvements) + 1
    
    print(f"  Best improvement at layer {best_layer}: +{np.max(layer_improvements):.3f}")
    print(f"  Worst improvement at layer {worst_layer}: +{np.min(layer_improvements):.3f}")
    
    # Relative improvement 
    if avg_raw > 0:
        relative_improvement = overall_improvement / (1 - avg_raw) * 100  # Percent of remaining distance to perfect
        print(f"  Relative improvement: {relative_improvement:.1f}% closer to perfect alignment")
    
    # Trend analysis
    print(f"\nTrend analysis:")
    early_layers = layer_improvements[:3]  # layers 1-3
    mid_layers = layer_improvements[3:7]   # layers 4-7  
    late_layers = layer_improvements[7:]   # layers 8-11
    
    print(f"  Early layers (1-3): +{np.mean(early_layers):.3f} average improvement")
    print(f"  Mid layers (4-7):   +{np.mean(mid_layers):.3f} average improvement")
    print(f"  Late layers (8-11): +{np.mean(late_layers):.3f} average improvement")
    
    # Alignment quality assessment
    print(f"\nAlignment quality:")
    if avg_trans > 0.8:
        quality = "very strong"
    elif avg_trans > 0.6:
        quality = "strong"
    elif avg_trans > 0.4:
        quality = "moderate"
    elif avg_trans > 0.2:
        quality = "weak"
    else:
        quality = "very weak"
    
    print(f"  Transformed token alignment: {quality} ({avg_trans:.3f} average)")
    print(f"  Raw token alignment: {avg_raw:.3f} average")
    
    # Show example tokens for context
    print(f"\nExample top-1 tokens from first prompt:")
    if len(results['all_top1_tokens']) > 0:
        example_tokens = results['all_top1_tokens'][0]  # First prompt
        for i, (layer, token) in enumerate(zip(results['layers'][:6], example_tokens[:6])):
            print(f"  Layer {layer}: '{token}'")
        if len(example_tokens) > 6:
            print(f"  ... (showing first 6 layers)")


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
    
    # Run token cosine similarity experiment
    results = run_multi_prompt_experiment(model, prompts, device)
    
    # Create poster plot
    save_path = create_token_cosine_plot(results)
    print(f"\nToken cosine similarity poster figure saved as: {save_path}")
    
    # Print detailed results
    print_token_cosine_results(results)


if __name__ == "__main__":
    main()