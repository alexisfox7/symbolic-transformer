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


def run_poster_experiment(model, text, device='cuda', k=3):
    """Run the main experiment for poster: Raw vs First-Layer embeddings across layers."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    n_layers = 12
    n_positions = len(tokens)
    
    # Store results - exclude layer 0
    layers = list(range(1, n_layers))
    raw_errors = []
    transformed_errors = []
    
    print(f"Running experiment on text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Using top-{k} tokens for reconstruction")
    print("=" * 60)
    
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
        
        print(f"Layer {layer_idx:2d}: Raw={avg_raw_error:5.1f}%, Transformed={avg_trans_error:5.1f}%, "
              f"Improvement={avg_raw_error-avg_trans_error:+5.1f}pp")
    
    return layers, raw_errors, transformed_errors, tokens


def create_poster_plot(layers, raw_errors, transformed_errors, text, save_path='poster_figure.png'):
    """Create a clean plot suitable for a poster."""
    plt.figure(figsize=(12, 5))  # More compressed height and wider
    
    # Plot lines with thicker lines and larger markers
    plt.plot(layers, raw_errors, 'o-', linewidth=4, markersize=12, 
             label='Raw Token Embeddings', color='#e74c3c')
    plt.plot(layers, transformed_errors, 's-', linewidth=4, markersize=12, 
             label='First-Layer Transformed', color='#3498db')
    
    # Calculate improvement across all layers (since we already excluded layer 0)
    improvement = np.mean(raw_errors) - np.mean(transformed_errors)
    
    # Formatting for poster - much larger text for visibility
    plt.xlabel('Layer', fontsize=24, fontweight='bold')
    plt.ylabel('Reconstruction Error (%)', fontsize=24, fontweight='bold')
    plt.title(f'Token Embedding Reconstruction Error by Layer\nText: "{text}" | Average improvement (layers 1-11): {improvement:.1f} percentage points', 
              fontsize=20, fontweight='bold', pad=20)
    
    plt.legend(fontsize=20, loc='center right')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers, fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def print_summary_stats(layers, raw_errors, transformed_errors):
    """Print summary statistics for the poster."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR POSTER")
    print("=" * 60)
    
    avg_raw = np.mean(raw_errors)
    avg_trans = np.mean(transformed_errors)
    improvement = avg_raw - avg_trans
    
    print(f"Average reconstruction error across all layers:")
    print(f"  Raw embeddings:           {avg_raw:.1f}%")
    print(f"  First-layer transformed:  {avg_trans:.1f}%")
    print(f"  Improvement:              {improvement:.1f} percentage points")
    
    # Find best and worst layers
    improvements_by_layer = np.array(raw_errors) - np.array(transformed_errors)
    best_layer = np.argmax(improvements_by_layer)
    worst_layer = np.argmin(improvements_by_layer)
    
    print(f"\nLayer-wise analysis:")
    print(f"  Best improvement at layer {best_layer}: {improvements_by_layer[best_layer]:.1f}pp")
    print(f"  Worst improvement at layer {worst_layer}: {improvements_by_layer[worst_layer]:.1f}pp")
    
    # Count layers where transformation helps
    helps_count = np.sum(improvements_by_layer > 0)
    print(f"  Transformation helps in {helps_count}/{len(layers)} layers")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Experiment settings
    text = "The girl was named"  # You can change this
    k = 3  # Number of top tokens to use
    
    # Run experiment
    layers, raw_errors, transformed_errors, tokens = run_poster_experiment(
        model, text, device, k=k
    )
    
    # Create poster-ready plot
    save_path = create_poster_plot(layers, raw_errors, transformed_errors, text)
    print(f"\nPoster figure saved as: {save_path}")
    
    # Print summary statistics
    print_summary_stats(layers, raw_errors, transformed_errors)


if __name__ == "__main__":
    main()