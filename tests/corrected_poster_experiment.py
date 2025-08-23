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
        # BUT: For layer 0, we should NOT use first-layer transformation 
        # since we're reconstructing the input to layer 0!
        if layer_idx == 0:
            # For layer 0, use raw embeddings for both methods
            transformed_embeddings = raw_embeddings
        else:
            # For layers 1+, use first-layer transformed embeddings
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
        
        # Transformed embeddings (same as raw for layer 0)
        A_trans = transformed_embeddings.T
        coeffs_trans = torch.linalg.lstsq(A_trans, b).solution.squeeze()
        recon_trans = (A_trans @ coeffs_trans.unsqueeze(1)).squeeze()
        error_trans = torch.norm(residual - recon_trans).item()
        
        return {
            'error_raw': error_raw,
            'error_transformed': error_trans,
            'residual_norm': torch.norm(residual).item(),
            'is_layer_zero': layer_idx == 0
        }


def run_experiment(model, text, device='cuda', k=3):
    """Run experiment across layers."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Test layers 0 through 11
    layers = list(range(12))
    n_positions = len(tokens)
    
    # Store results
    raw_errors = []
    transformed_errors = []
    
    print(f"Running experiment on text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Using top-{k} tokens for reconstruction")
    print("Note: Layer 0 uses raw embeddings for both methods (no first-layer transform)")
    print("=" * 70)
    
    for layer_idx in layers:
        layer_raw_errors = []
        layer_trans_errors = []
        
        print(f"Processing layer {layer_idx}...")
        
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
        
        if layer_idx == 0:
            print(f"Layer {layer_idx:2d}: Raw={avg_raw_error:5.1f}%, Transformed={avg_trans_error:5.1f}% "
                  f"(same - no transform for layer 0)")
        else:
            print(f"Layer {layer_idx:2d}: Raw={avg_raw_error:5.1f}%, Transformed={avg_trans_error:5.1f}%, "
                  f"Improvement={avg_raw_error-avg_trans_error:+5.1f}pp")
    
    return layers, raw_errors, transformed_errors, tokens


def create_poster_plot(layers, raw_errors, transformed_errors, text, save_path='poster_figure.png'):
    """Create a clean plot suitable for a poster."""
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(layers, raw_errors, 'o-', linewidth=2.5, markersize=8, 
             label='Raw Token Embeddings', color='#e74c3c')
    plt.plot(layers, transformed_errors, 's-', linewidth=2.5, markersize=8, 
             label='First-Layer Transformed', color='#3498db')
    
    # Formatting for poster
    plt.xlabel('Layer', fontsize=16)
    plt.ylabel('Reconstruction Error (%)', fontsize=16)
    plt.title(f'Token Embedding Reconstruction Error by Layer\nText: "{text}"', 
              fontsize=18, pad=20)
    
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add improvement annotation (excluding layer 0)
    improvement = np.mean(np.array(raw_errors)[1:]) - np.mean(np.array(transformed_errors)[1:])
    plt.text(0.02, 0.98, f'Average improvement (layers 1-11): {improvement:.1f} percentage points', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Experiment settings
    text = "The cat sat"  # Shorter for speed
    k = 3  # Number of top tokens to use
    
    # Run experiment
    layers, raw_errors, transformed_errors, tokens = run_experiment(
        model, text, device, k=k
    )
    
    # Create poster-ready plot
    save_path = create_poster_plot(layers, raw_errors, transformed_errors, text)
    print(f"\nPoster figure saved as: {save_path}")
    
    # Print summary (excluding layer 0 where methods are identical)
    print("\n" + "=" * 70)
    print("SUMMARY FOR POSTER")
    print("=" * 70)
    
    # Calculate averages excluding layer 0
    avg_raw = np.mean(raw_errors[1:])  # layers 1-11
    avg_trans = np.mean(transformed_errors[1:])  # layers 1-11
    improvement = avg_raw - avg_trans
    
    print(f"Average reconstruction error (layers 1-11):")
    print(f"  Raw embeddings:           {avg_raw:.1f}%")
    print(f"  First-layer transformed:  {avg_trans:.1f}%")
    print(f"  Improvement:              {improvement:.1f} percentage points")
    
    print(f"\nLayer 0 note: Both methods identical ({raw_errors[0]:.1f}%) - no transformation applied")


if __name__ == "__main__":
    main()