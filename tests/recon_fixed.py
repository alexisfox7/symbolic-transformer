import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def analyze_top1_reconstruction(model, input_ids, layer_idx, position_idx):
    """
    Analyze reconstruction using only top-1 token with detailed diagnostics.
    """
    device = input_ids.device
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get residual at position
        if layer_idx == -1:
            residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top token via logit lens
        residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        
        # Get top 1 token
        top_value, top_idx = torch.max(logits[0], dim=0)
        top_embedding = model.transformer.wte(top_idx.unsqueeze(0))[0]  # [d_model]
        
        # Calculate metrics
        residual_norm = torch.norm(residual).item()
        embedding_norm = torch.norm(top_embedding).item()
        
        # Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            residual.unsqueeze(0), 
            top_embedding.unsqueeze(0)
        ).item()
        
        # L2 distance
        l2_distance = torch.norm(residual - top_embedding).item()
        
        # Normalized L2 (as percentage of residual norm)
        normalized_l2 = (l2_distance / residual_norm * 100) if residual_norm > 0 else 0
        
        # Decode token
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        token_str = tokenizer.decode([top_idx.item()])
        
        return {
            'token': token_str,
            'token_id': top_idx.item(),
            'logit': top_value.item(),
            'residual_norm': residual_norm,
            'embedding_norm': embedding_norm,
            'cosine_similarity': cosine_sim,
            'l2_distance': l2_distance,
            'normalized_l2': normalized_l2,
            'residual': residual,
            'embedding': top_embedding
        }


def comprehensive_analysis(model, text, device='cuda'):
    """
    Comprehensive analysis of top-1 token reconstruction.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    n_layers = model.config.n_layer
    n_positions = input_ids.shape[1]
    
    # Store results
    results = {
        'cosine_sim': np.zeros((n_layers + 1, n_positions)),
        'l2_distance': np.zeros((n_layers + 1, n_positions)),
        'normalized_l2': np.zeros((n_layers + 1, n_positions))
    }
    
    print(f"Analyzing: {text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    
    for layer_idx in range(-1, n_layers):
        print(f"\nLayer {layer_idx:2d}:")
        for pos_idx in range(n_positions):
            res = analyze_top1_reconstruction(model, input_ids, layer_idx, pos_idx)
            
            results['cosine_sim'][layer_idx + 1, pos_idx] = res['cosine_similarity']
            results['l2_distance'][layer_idx + 1, pos_idx] = res['l2_distance']
            results['normalized_l2'][layer_idx + 1, pos_idx] = res['normalized_l2']
            
            print(f"  Pos {pos_idx} ({tokens[pos_idx]:10s}): "
                  f"Top='{res['token']:10s}' "
                  f"Cosine={res['cosine_similarity']:6.3f} "
                  f"L2={res['l2_distance']:7.2f} "
                  f"Norm%={res['normalized_l2']:6.1f}% "
                  f"Logit={res['logit']:7.2f}")
    
    return results


def plot_similarity_analysis(results, text):
    """
    Visualize cosine similarity and L2 distance.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cosine similarity heatmap
    ax = axes[0, 0]
    im = ax.imshow(results['cosine_sim'], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Cosine Similarity (Residual vs Top-1 Embedding)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im, ax=ax)
    
    # L2 distance heatmap
    ax = axes[0, 1]
    im = ax.imshow(results['l2_distance'], aspect='auto', cmap='viridis')
    ax.set_title('L2 Distance')
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im, ax=ax)
    
    # Average cosine similarity by layer
    ax = axes[1, 0]
    avg_cosine = results['cosine_sim'].mean(axis=1)
    ax.plot(range(-1, len(avg_cosine)-1), avg_cosine, marker='o')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Cosine Similarity')
    ax.set_title('Average Cosine Similarity by Layer')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Normalized L2 by layer
    ax = axes[1, 1]
    avg_norm_l2 = results['normalized_l2'].mean(axis=1)
    ax.plot(range(-1, len(avg_norm_l2)-1), avg_norm_l2, marker='o', color='orange')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized L2 Distance (%)')
    ax.set_title('Average Normalized L2 Distance by Layer')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% (orthogonal)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


def check_embedding_space(model, text, device='cuda'):
    """
    Check if embeddings and residuals are in same space.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print("\n" + "=" * 80)
    print("Checking embedding space alignment:")
    print("=" * 80)
    
    with torch.no_grad():
        # Get initial embeddings
        embeddings = model.transformer.wte(input_ids[0])  # [seq_len, d_model]
        
        # Get residuals at layer 0 (after embedding + positional)
        outputs = model(input_ids, output_hidden_states=True)
        layer0_residuals = outputs.hidden_states[0][0]  # [seq_len, d_model]
        
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Layer 0 residual shape: {layer0_residuals.shape}")
        
        # Check if layer -1 (pure embeddings) matches input embeddings
        for i in range(min(3, input_ids.shape[1])):
            token_id = input_ids[0, i].item()
            direct_embedding = model.transformer.wte(input_ids[0, i].unsqueeze(0))[0]
            
            # Compare with what we get from hidden_states
            hidden_state_embedding = outputs.hidden_states[0][0, i]
            
            diff = torch.norm(direct_embedding - embeddings[i]).item()
            print(f"\nPosition {i} (token_id={token_id}):")
            print(f"  Direct embedding norm: {torch.norm(direct_embedding).item():.3f}")
            print(f"  Hidden state[0] norm: {torch.norm(hidden_state_embedding).item():.3f}")
            print(f"  Difference: {diff:.6f}")
            
            # Check positional encoding
            pos_encoding = model.transformer.wpe(torch.tensor([i], device=device))[0]
            print(f"  Positional encoding norm: {torch.norm(pos_encoding).item():.3f}")
            
            # Reconstruct layer 0 hidden state
            reconstructed = direct_embedding + pos_encoding
            layer0_diff = torch.norm(reconstructed - hidden_state_embedding).item()
            print(f"  Embedding + Positional vs Hidden[0]: diff={layer0_diff:.6f}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox"
    
    # First check embedding space alignment
    check_embedding_space(model, text, device)
    
    print("\n" + "=" * 80)
    print("TOP-1 TOKEN RECONSTRUCTION ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = comprehensive_analysis(model, text, device)
    
    # Plot results
    print("\n" + "=" * 80)
    print("Plotting results...")
    plot_similarity_analysis(results, text)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    print(f"Average cosine similarity across all layers: {results['cosine_sim'].mean():.3f}")
    print(f"Average normalized L2 distance: {results['normalized_l2'].mean():.1f}%")
    print(f"Cosine similarity at embedding layer: {results['cosine_sim'][0].mean():.3f}")
    print(f"Cosine similarity at final layer: {results['cosine_sim'][-1].mean():.3f}")
    
    return results


if __name__ == "__main__":
    results = main()