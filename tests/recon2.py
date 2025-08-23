import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def get_residual_and_decompose_lstsq(model, input_ids, layer_idx, position_idx, k=3):
    """
    Decompose residual using least squares fitting to top-k token embeddings.
    """
    device = input_ids.device
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        if layer_idx == -1:
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via logit lens
        residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        topk_values, topk_indices = torch.topk(logits[0], k=k)
        
        # Get embeddings for top k tokens
        topk_embeddings = model.transformer.wte(topk_indices)  # [k, d_model]
        
        # Solve least squares: find coefficients α such that Σ(α_i * embedding_i) ≈ residual
        # This is solving: E @ alpha = r, where E is [d_model, k] and r is [d_model]
        E = topk_embeddings.T  # [d_model, k]
        r = residual_at_pos.unsqueeze(1)  # [d_model, 1]
        
        # Least squares solution
        alpha = torch.linalg.lstsq(E, r).solution  # [k, 1]
        alpha = alpha.squeeze()
        
        # Reconstruct using fitted coefficients
        reconstruction = (topk_embeddings.T @ alpha.unsqueeze(1)).squeeze()
        
        # Calculate components
        decomposed = {
            'original': residual_at_pos,
            'reconstruction': reconstruction,
            'coefficients': alpha,
            'topk_embeddings': topk_embeddings,
            'dark_matter': residual_at_pos - reconstruction
        }
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        token_names = [tokenizer.decode([idx.item()]) for idx in topk_indices]
        
        return decomposed, token_names, topk_values


def analyze_reconstruction_methods(model, text, device='cuda'):
    """
    Compare different reconstruction methods.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    n_layers = model.config.n_layer
    n_positions = input_ids.shape[1]
    
    results = {
        'simple_sum': np.zeros((n_layers + 1, n_positions)),
        'lstsq': np.zeros((n_layers + 1, n_positions)),
        'weighted_sum': np.zeros((n_layers + 1, n_positions))
    }
    
    print("Comparing reconstruction methods at Layer 5, Position 4:")
    print("=" * 60)
    
    for layer_idx in range(-1, n_layers):
        for pos_idx in range(n_positions):
            # Method 1: Simple sum (your original)
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                if layer_idx == -1:
                    residual = outputs.hidden_states[0][0, pos_idx, :]
                else:
                    residual = outputs.hidden_states[layer_idx][0, pos_idx, :]
                
                residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
                logits = model.lm_head(residual_normed)
                top3_indices = torch.topk(logits[0], k=3).indices
                top3_embeddings = model.transformer.wte(top3_indices)
                
                simple_recon = top3_embeddings.sum(dim=0)
                simple_loss = torch.norm(residual - simple_recon).item()
                results['simple_sum'][layer_idx + 1, pos_idx] = simple_loss
            
            # Method 2: Least squares
            decomposed, tokens, logit_values = get_residual_and_decompose_lstsq(
                model, input_ids, layer_idx, pos_idx, k=3
            )
            lstsq_loss = torch.norm(decomposed['dark_matter']).item()
            results['lstsq'][layer_idx + 1, pos_idx] = lstsq_loss
            
            # Method 3: Weighted by softmax of logits
            with torch.no_grad():
                weights = torch.softmax(logit_values, dim=0)
                weighted_recon = (top3_embeddings.T @ weights.unsqueeze(1)).squeeze()
                weighted_loss = torch.norm(residual - weighted_recon).item()
                results['weighted_sum'][layer_idx + 1, pos_idx] = weighted_loss
            
            # Print detailed comparison for one specific position
            if layer_idx == 5 and pos_idx == 4:
                orig_norm = torch.norm(residual).item()
                print(f"\nOriginal norm: {orig_norm:.3f}")
                print(f"Simple sum loss: {simple_loss:.3f} ({100*simple_loss/orig_norm:.1f}%)")
                print(f"Least squares loss: {lstsq_loss:.3f} ({100*lstsq_loss/orig_norm:.1f}%)")
                print(f"Weighted sum loss: {weighted_loss:.3f} ({100*weighted_loss/orig_norm:.1f}%)")
                print(f"Top tokens: {tokens}")
                print(f"Least squares coefficients: {decomposed['coefficients'].cpu().numpy()}")
                print(f"Softmax weights: {weights.cpu().numpy()}")
    
    return results


def plot_method_comparison(results, text):
    """
    Compare reconstruction methods visually.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot heatmaps for each method
    methods = ['simple_sum', 'lstsq', 'weighted_sum']
    titles = ['Simple Sum', 'Least Squares', 'Weighted Sum']
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(results[method], aspect='auto', cmap='viridis')
        ax.set_title(f'{title} Reconstruction Loss')
        ax.set_xlabel('Position')
        ax.set_ylabel('Layer')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        plt.colorbar(im, ax=ax)
    
    # Compare average losses
    ax = axes[1, 1]
    for method, title in zip(methods, titles):
        avg_loss = results[method].mean(axis=1)
        ax.plot(range(-1, len(avg_loss)-1), avg_loss, marker='o', label=title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Reconstruction Loss')
    ax.set_title('Method Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_coefficient_patterns(model, text, device='cuda'):
    """
    Analyze the least squares coefficients across layers.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    n_layers = model.config.n_layer
    selected_pos = len(input_ids[0]) // 2  # Middle position
    
    print(f"\nAnalyzing coefficients at position {selected_pos}:")
    print("=" * 60)
    
    for layer_idx in [0, 3, 6, 9, 11]:  # Sample layers
        decomposed, tokens, _ = get_residual_and_decompose_lstsq(
            model, input_ids, layer_idx, selected_pos, k=5
        )
        
        coeffs = decomposed['coefficients'].cpu().numpy()
        orig_norm = torch.norm(decomposed['original']).item()
        recon_norm = torch.norm(decomposed['reconstruction']).item()
        error_norm = torch.norm(decomposed['dark_matter']).item()
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Original norm: {orig_norm:.3f}")
        print(f"  Reconstruction norm: {recon_norm:.3f}")
        print(f"  Error norm: {error_norm:.3f} ({100*error_norm/orig_norm:.1f}%)")
        print(f"  Top 5 tokens and coefficients:")
        for token, coeff in zip(tokens, coeffs):
            print(f"    {token:15s}: {coeff:7.3f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox jumps over the lazy dog"
    
    print("Comparing reconstruction methods...")
    results = analyze_reconstruction_methods(model, text, device)
    
    print("\n" + "=" * 60)
    print("Plotting comparison...")
    plot_method_comparison(results, text)
    
    print("\n" + "=" * 60)
    analyze_coefficient_patterns(model, text, device)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    for method in results:
        avg_loss = results[method].mean()
        print(f"{method:15s}: Average loss = {avg_loss:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()