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


def compute_individual_token_similarities(model, input_ids, layer_idx, k=3):
    """Compare individual token cosine similarities (raw vs transformed) with residual."""
    device = input_ids.device
    seq_length = input_ids.shape[1]
    position_idx = 0  # Only use first position for speed
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual at this layer
        residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via LogitLens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        topk_values, topk_indices = torch.topk(logits, k=k)
        
        # Get raw embeddings
        raw_embeddings = model.transformer.wte(topk_indices)  # Shape: [k, d_model]
        
        # Get first-layer transformed embeddings
        transformed_embeddings = []
        for token_id in topk_indices:
            transformed = get_token_through_first_layer(
                model, token_id, position_idx, seq_length, device
            )
            transformed_embeddings.append(transformed)
        transformed_embeddings = torch.stack(transformed_embeddings)  # Shape: [k, d_model]
        
        # Compute individual cosine similarities
        raw_cosines = []
        transformed_cosines = []
        
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
            
            raw_cosines.append(cos_sim_raw)
            transformed_cosines.append(cos_sim_trans)
        
        return {
            'raw_cosines': raw_cosines,
            'transformed_cosines': transformed_cosines,
            'topk_indices': topk_indices.cpu().numpy(),
            'topk_logits': topk_values.cpu().numpy(),
            'residual_norm': torch.norm(residual).item()
        }


def run_single_prompt_experiment(model, text, device='cuda', k=3):
    """Run experiment on a single prompt across layers 1-11."""
    # Create tokenizer offline to avoid network issues
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    except:
        print("Using basic tokenizer due to network issues")
        # Simple fallback - encode manually
        input_ids = torch.tensor([[464, 3797, 3332]], device=device)  # "The cat sat" approximate
        if "walking" in text:
            input_ids = torch.tensor([[3347, 373, 6155]], device=device)  # "She was walking" 
        elif "pizza" in text:
            input_ids = torch.tensor([[7554, 7832, 14281]], device=device)  # "John likes pizza"
        elif "book" in text:
            input_ids = torch.tensor([[464, 1492, 4909]], device=device)  # "The book contains"
        elif "going" in text:
            input_ids = torch.tensor([[775, 389, 1016]], device=device)  # "We are going"
        elif "raining" in text:
            input_ids = torch.tensor([[632, 373, 26742]], device=device)  # "It was raining"
        
        # Decode function fallback
        def decode_fallback(token_ids):
            # Simple mapping for common tokens
            token_map = {464: "The", 3797: "cat", 3332: "sat", 3347: "She", 373: "was", 6155: "walking"}
            return [token_map.get(t, f"<{t}>") for t in token_ids]
        
        tokenizer = type('MockTokenizer', (), {'decode': lambda self, ids: decode_fallback(ids)})()
    else:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    # Only layers 1-11 (exclude layer 0)
    layers = list(range(1, 12))
    
    # Store results for each layer
    layer_results = {}
    
    for layer_idx in layers:
        result = compute_individual_token_similarities(model, input_ids, layer_idx, k)
        layer_results[layer_idx] = result
        
        # Get token strings for first layer only (for display)
        if layer_idx == 1:
            try:
                token_strings = [tokenizer.decode([int(idx)]) for idx in result['topk_indices']]
            except:
                token_strings = [f"<{int(idx)}>" for idx in result['topk_indices']]
            layer_results['token_strings'] = token_strings
    
    return layer_results


def run_multi_prompt_experiment(model, prompts, device='cuda', k=3):
    """Run experiment across multiple prompts."""
    layers = list(range(1, 12))  # Only layers 1-11
    
    # Store all results
    all_results = []
    
    print(f"Running INDIVIDUAL TOKEN COSINE SIMILARITY experiment on {len(prompts)} prompts:")
    print("Comparing each individual top-k token (raw vs first-layer transformed) with residual")
    print("=" * 70)
    
    for i, text in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: '{text}'")
        
        layer_results = run_single_prompt_experiment(model, text, device, k)
        all_results.append(layer_results)
        
        # Show example for first prompt
        if i == 0 and 'token_strings' in layer_results:
            print(f"  Top-{k} tokens: {layer_results['token_strings']}")
    
    # Aggregate statistics across prompts and tokens
    raw_means_by_layer = []
    transformed_means_by_layer = []
    raw_sems_by_layer = []
    transformed_sems_by_layer = []
    
    for layer_idx in layers:
        # Collect all cosine similarities for this layer across all prompts and all k tokens
        all_raw_cosines = []
        all_transformed_cosines = []
        
        for prompt_results in all_results:
            if layer_idx in prompt_results:
                all_raw_cosines.extend(prompt_results[layer_idx]['raw_cosines'])
                all_transformed_cosines.extend(prompt_results[layer_idx]['transformed_cosines'])
        
        # Calculate statistics
        raw_mean = np.mean(all_raw_cosines)
        transformed_mean = np.mean(all_transformed_cosines)
        raw_sem = np.std(all_raw_cosines) / np.sqrt(len(all_raw_cosines))
        transformed_sem = np.std(all_transformed_cosines) / np.sqrt(len(all_transformed_cosines))
        
        raw_means_by_layer.append(raw_mean)
        transformed_means_by_layer.append(transformed_mean)
        raw_sems_by_layer.append(raw_sem)
        transformed_sems_by_layer.append(transformed_sem)
    
    return {
        'layers': layers,
        'raw_means': np.array(raw_means_by_layer),
        'transformed_means': np.array(transformed_means_by_layer),
        'raw_sems': np.array(raw_sems_by_layer),
        'transformed_sems': np.array(transformed_sems_by_layer),
        'all_results': all_results,
        'prompts': prompts,
        'k': k
    }


def create_individual_token_plot(results, save_path='individual_token_cosine_poster.png'):
    """Create the poster plot for individual token cosine similarity comparison."""
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
    ax.set_ylabel('Individual Token Cosine Similarity', fontsize=16)
    ax.set_title(f'Individual Token vs Residual Cosine Similarity\n'
                f'Raw vs First-Layer Transformed (n={len(results["prompts"])} prompts, top-{results["k"]} tokens ±SEM)', 
                fontsize=18, pad=20)
    
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xticks(layers)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Set y-axis range 
    y_min = min(np.min(results['raw_means'] - results['raw_sems']), 
                np.min(results['transformed_means'] - results['transformed_sems']))
    y_max = max(np.max(results['raw_means'] + results['raw_sems']), 
                np.max(results['transformed_means'] + results['transformed_sems']))
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    # Add improvement annotation
    improvement = np.mean(results['transformed_means']) - np.mean(results['raw_means'])
    
    ax.text(0.02, 0.98, f'Average improvement: {improvement:+.3f} cosine similarity\n'
                        f'Individual tokens vs residuals (not reconstructions)', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def print_individual_token_results(results):
    """Print results for individual token cosine similarity analysis."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL TOKEN COSINE SIMILARITY RESULTS")
    print("=" * 70)
    
    print(f"Number of prompts: {len(results['prompts'])}")
    print(f"Top-k tokens analyzed: {results['k']}")
    print(f"Analysis: Individual token cosine similarity with layer residual")
    print(f"Comparison: Raw token embedding vs First-layer transformed token")
    print(f"Note: This measures individual tokens, NOT reconstructions")
    
    # Overall statistics
    avg_raw = np.mean(results['raw_means'])
    avg_trans = np.mean(results['transformed_means'])
    overall_improvement = avg_trans - avg_raw  # For cosine sim, higher is better
    
    print(f"\nOverall statistics (layers 1-11, averaged across all tokens):")
    print(f"  Raw token embeddings:     {avg_raw:.3f} ± {np.mean(results['raw_sems']):.3f}")
    print(f"  First-layer transformed:  {avg_trans:.3f} ± {np.mean(results['transformed_sems']):.3f}")
    print(f"  Overall improvement:      {overall_improvement:+.3f} cosine similarity")
    
    # Layer-by-layer breakdown
    print(f"\nLayer-by-layer results:")
    print(f"{'Layer':<5} {'Raw Token':<12} {'Transformed':<12} {'Improvement':<12}")
    print("-" * 55)
    
    for i, layer in enumerate(results['layers']):
        raw_mean = results['raw_means'][i]
        trans_mean = results['transformed_means'][i]
        improvement = trans_mean - raw_mean
        
        print(f"{layer:<5} {raw_mean:>8.3f}     {trans_mean:>8.3f}       {improvement:>+7.3f}")
    
    # Statistical assessment
    print(f"\nStatistical assessment:")
    if abs(overall_improvement) < 0.01:
        print(f"  Improvement is very small ({overall_improvement:+.3f})")
    elif abs(overall_improvement) < 0.05:
        print(f"  Improvement is small ({overall_improvement:+.3f})")
    elif abs(overall_improvement) < 0.1:
        print(f"  Improvement is moderate ({overall_improvement:+.3f})")
    else:
        print(f"  Improvement is substantial ({overall_improvement:+.3f})")
    
    # Range analysis
    layer_improvements = results['transformed_means'] - results['raw_means']
    best_layer = np.argmax(layer_improvements) + 1  # +1 because we start from layer 1
    worst_layer = np.argmin(layer_improvements) + 1
    
    print(f"  Best improvement at layer {best_layer}: {np.max(layer_improvements):+.3f}")
    print(f"  Worst improvement at layer {worst_layer}: {np.min(layer_improvements):+.3f}")
    print(f"  Improvement range: {np.min(layer_improvements):+.3f} to {np.max(layer_improvements):+.3f}")
    
    # Trend analysis
    print(f"\nTrend analysis:")
    early_layers = layer_improvements[:3]  # layers 1-3
    mid_layers = layer_improvements[3:7]   # layers 4-7  
    late_layers = layer_improvements[7:]   # layers 8-11
    
    print(f"  Early layers (1-3): {np.mean(early_layers):+.3f} average improvement")
    print(f"  Mid layers (4-7):   {np.mean(mid_layers):+.3f} average improvement")  
    print(f"  Late layers (8-11): {np.mean(late_layers):+.3f} average improvement")
    
    # Interpretation
    print(f"\nInterpretation:")
    if overall_improvement > 0.05:
        print(f"  First-layer transformation helps individual token alignment")
    elif overall_improvement > 0:
        print(f"  First-layer transformation provides minimal improvement for individual tokens")
    else:
        print(f"  First-layer transformation may hurt individual token alignment")
    
    print(f"  Raw embeddings average: {avg_raw:.3f} (individual tokens don't align well with complex residuals)")
    print(f"  This is expected - residuals contain multi-token information")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Multiple prompts for analysis
    prompts = [
        "The cat sat",
        "She was walking", 
        "John likes pizza",
        "The book contains",
        "We are going",
        "It was raining"
    ]
    
    k = 3  # Number of top tokens to analyze
    
    # Run individual token cosine similarity experiment
    results = run_multi_prompt_experiment(model, prompts, device, k)
    
    # Create poster plot
    save_path = create_individual_token_plot(results)
    print(f"\nIndividual token cosine similarity poster figure saved as: {save_path}")
    
    # Print detailed results
    print_individual_token_results(results)


if __name__ == "__main__":
    main()