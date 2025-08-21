import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx, tokenizer):
    """
    Simpler version using HuggingFace's output_hidden_states.
    layer_idx: -1 for embeddings, 0 to n_layers-1 for layer outputs
    """
    device = input_ids.device
    
    # Validate inputs
    if position_idx >= input_ids.shape[1]:
        raise ValueError(f"position_idx {position_idx} out of range for sequence length {input_ids.shape[1]}")
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # hidden_states includes embeddings as layer 0, so layer_idx+1
        if layer_idx == -1:
            # Initial embeddings (before any transformer layers)
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            # After layer layer_idx-1 (hidden_states[0] is embeddings, so layer 0 output is at index 1)
            residual_at_pos = outputs.hidden_states[layer_idx + 1][0, position_idx, :]
        
        # Apply layer normalization for LogitLens (all layers need ln_f for vocab projection)
        residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        
        # Get top 3
        top3_indices = torch.topk(logits[0], k=3).indices
        top3_embeddings = model.transformer.wte(top3_indices)
        
        decomposed = {
            'original': residual_at_pos,
            'word1': top3_embeddings[0],
            'word2': top3_embeddings[1],
            'word3': top3_embeddings[2],
            'dark_matter': residual_at_pos - top3_embeddings.sum(dim=0)
        }
        
        token_names = [tokenizer.decode(idx.item()) for idx in top3_indices]
        
        return decomposed, token_names


def analyze_reconstruction_loss(model, text, tokenizer, device='cuda'):
    """
    Analyze reconstruction loss across all layers and positions.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    n_layers = model.config.n_layer
    n_positions = input_ids.shape[1]
    
    # Store losses for each layer and position
    losses = np.zeros((n_layers + 1, n_positions))  # +1 for embedding layer
    
    print(f"Analyzing {n_positions} positions across {n_layers + 1} layers (including embeddings)")
    print(f"Input text: '{text}'")
    tokens_repr = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print(f"Input tokens: {tokens_repr}")
    print("=" * 60)
    
    for layer_idx in range(-1, n_layers):
        for pos_idx in range(n_positions):
            decomposed, tokens = get_residual_and_decompose_simple(
                model, input_ids, layer_idx, pos_idx, tokenizer
            )
            
            # Reconstruct using sum of top 3 embeddings
            reconstruction = decomposed['word1'] + decomposed['word2'] + decomposed['word3']
            original = decomposed['original']
            
            # Calculate L2 reconstruction loss
            loss = torch.norm(original - reconstruction).item()
            losses[layer_idx + 1, pos_idx] = loss
            
            # Calculate normalized loss (as percentage of original norm)
            orig_norm = torch.norm(original).item()
            if orig_norm > 0:
                normalized_loss = loss / orig_norm * 100
            else:
                normalized_loss = 0
            
            # Clean up token strings for display
            clean_tokens = [repr(t) for t in tokens]
            print(f"Layer {layer_idx:2d}, Pos {pos_idx:2d}: Loss={loss:.3f}, "
                  f"Normalized={normalized_loss:.1f}%, Top tokens: {clean_tokens}")
    
    return losses


def plot_reconstruction_losses(losses, text, tokenizer):
    """
    Visualize reconstruction losses as a heatmap.
    """
    tokens = tokenizer.tokenize(text)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Heatmap of raw losses
    im1 = ax1.imshow(losses, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Layer')
    ax1.set_title('Reconstruction Loss (L2 norm)')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    plt.colorbar(im1, ax=ax1)
    
    # Average loss per layer
    avg_losses = losses.mean(axis=1)
    ax2.plot(range(-1, len(avg_losses)-1), avg_losses, marker='o')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Reconstruction Loss')
    ax2.set_title('Average Loss by Layer')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def analyze_dark_matter_ratio(model, text, tokenizer, device='cuda'):
    """
    Analyze the ratio of 'dark matter' (unexplained residual) across layers.
    """
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    n_layers = model.config.n_layer
    n_positions = input_ids.shape[1]
    
    dark_matter_ratios = []
    
    for layer_idx in range(-1, n_layers):
        layer_ratios = []
        for pos_idx in range(n_positions):
            decomposed, _ = get_residual_and_decompose_simple(
                model, input_ids, layer_idx, pos_idx, tokenizer
            )
            
            original_norm = torch.norm(decomposed['original']).item()
            dark_matter_norm = torch.norm(decomposed['dark_matter']).item()
            
            if original_norm > 0:
                ratio = dark_matter_norm / original_norm
                layer_ratios.append(ratio)
        
        avg_ratio = np.mean(layer_ratios)
        dark_matter_ratios.append(avg_ratio)
        print(f"Layer {layer_idx:2d}: Dark matter ratio = {avg_ratio:.3f}")
    
    return dark_matter_ratios


def main():
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Ensure we have padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog"
    
    print("Analyzing reconstruction loss...")
    print("=" * 60)
    losses = analyze_reconstruction_loss(model, text, tokenizer, device)
    
    print("\n" + "=" * 60)
    print("Plotting results...")
    plot_reconstruction_losses(losses, text, tokenizer)
    
    print("\n" + "=" * 60)
    print("Analyzing dark matter ratios...")
    dark_matter_ratios = analyze_dark_matter_ratio(model, text, tokenizer, device)
    
    # Plot dark matter ratios
    plt.figure(figsize=(10, 6))
    plt.plot(range(-1, len(dark_matter_ratios)-1), dark_matter_ratios, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Dark Matter Ratio')
    plt.title('Proportion of Residual Not Explained by Top 3 Tokens')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return losses, dark_matter_ratios


if __name__ == "__main__":
    losses, dark_matter_ratios = main()