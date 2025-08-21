import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

class ReversibleLayerNorm:
    """
    A wrapper that stores statistics to make layer norm reversible.
    """
    def __init__(self, ln_module, eps=1e-5):
        self.ln_module = ln_module
        self.eps = eps
        self.stored_mean = None
        self.stored_std = None
        
    def forward(self, x):
        """
        Apply layer norm while storing statistics.
        """
        # Store original statistics
        self.stored_mean = x.mean(dim=-1, keepdim=True)
        self.stored_std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        
        # Apply layer norm
        output = self.ln_module(x)
        
        return output
    
    def reverse(self, normed_output):
        """
        Reverse the layer norm using stored statistics.
        """
        if self.stored_mean is None or self.stored_std is None:
            raise ValueError("No statistics stored! Run forward() first.")
        
        # Remove affine transformation
        weight = self.ln_module.weight
        bias = self.ln_module.bias
        x_normalized = (normed_output - bias) / weight
        
        # Restore original scale and shift using stored statistics
        x_restored = x_normalized * self.stored_std + self.stored_mean
        
        return x_restored


def test_reversible_layer_norm(model, text, device='cuda'):
    """
    Test that we can perfectly reverse layer norm with stored statistics.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print(f"Testing reversible layer norm on: {text}")
    print("=" * 80)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get a residual stream vector
        layer_idx = 5
        pos_idx = 2
        residual = outputs.hidden_states[layer_idx][0, pos_idx, :]
        
        # Create reversible layer norm
        reversible_ln = ReversibleLayerNorm(model.transformer.ln_f)
        
        # Forward pass (stores statistics)
        normed = reversible_ln.forward(residual.unsqueeze(0))[0]
        
        # Reverse pass
        restored = reversible_ln.reverse(normed.unsqueeze(0))[0]
        
        # Check if perfect reconstruction
        reconstruction_error = torch.norm(residual - restored).item()
        
        print(f"Original residual norm: {torch.norm(residual).item():.6f}")
        print(f"After layer norm: {torch.norm(normed).item():.6f}")
        print(f"After reversal: {torch.norm(restored).item():.6f}")
        print(f"Reconstruction error: {reconstruction_error:.10f}")
        print(f"Perfect reconstruction: {reconstruction_error < 1e-5}")


def decompose_with_reversible_ln(model, input_ids, layer_idx, position_idx):
    """
    Decompose using LogitLens but with ability to reverse the layer norm.
    """
    device = input_ids.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get residual
        if layer_idx == -1:
            residual = outputs.hidden_states[0][0, position_idx, :]
        else:
            residual = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Create reversible layer norm
        reversible_ln = ReversibleLayerNorm(model.transformer.ln_f)
        
        # Apply layer norm (and store stats)
        residual_normed = reversible_ln.forward(residual.unsqueeze(0))[0]
        
        # Get top k tokens via LogitLens
        logits = model.lm_head(residual_normed.unsqueeze(0))[0]
        top_k = 3
        topk_values, topk_indices = torch.topk(logits, k=top_k)
        
        # Get embeddings for top k tokens
        topk_embeddings = model.transformer.wte(topk_indices)  # [k, d_model]
        
        # Now we can work in different spaces:
        
        # Option 1: Compare in normalized space
        topk_embeddings_normed = model.transformer.ln_f(topk_embeddings)
        
        # Option 2: Reverse the layer norm on residual to get back to original space
        residual_restored = reversible_ln.reverse(residual_normed.unsqueeze(0))[0]
        
        # Option 3: Find coefficients in the normalized space
        # Solve: minimize ||residual_normed - sum(alpha_i * embed_normed_i)||
        A = topk_embeddings_normed.T  # [d_model, k]
        b = residual_normed.unsqueeze(1)  # [d_model, 1]
        coeffs_normed = torch.linalg.lstsq(A, b).solution.squeeze()
        
        # Reconstruct in normalized space
        reconstruction_normed = (topk_embeddings_normed.T @ coeffs_normed.unsqueeze(1)).squeeze()
        
        # Reverse both to original space for comparison
        reconstruction_original = reversible_ln.reverse(reconstruction_normed.unsqueeze(0))[0]
        
        results = {
            'residual_original': residual,
            'residual_normed': residual_normed,
            'residual_restored': residual_restored,
            'topk_indices': topk_indices,
            'topk_tokens': [tokenizer.decode([idx.item()]) for idx in topk_indices],
            'coefficients_normed_space': coeffs_normed,
            'reconstruction_normed': reconstruction_normed,
            'reconstruction_original': reconstruction_original,
            'error_normed': torch.norm(residual_normed - reconstruction_normed).item(),
            'error_original': torch.norm(residual - reconstruction_original).item()
        }
        
        return results


def analyze_decomposition_spaces(model, text, device='cuda'):
    """
    Compare decomposition in original vs normalized space.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"\nAnalyzing decomposition in different spaces")
    print(f"Text: {text}")
    print("=" * 80)
    
    for layer_idx in [0, 3, 6, 9, 11]:
        print(f"\nLayer {layer_idx}:")
        print("-" * 40)
        
        for pos_idx in range(min(3, len(tokens))):
            results = decompose_with_reversible_ln(model, input_ids, layer_idx, pos_idx)
            
            print(f"\nPosition {pos_idx} ('{tokens[pos_idx]}'):")
            print(f"  Top 3 tokens: {results['topk_tokens']}")
            print(f"  Coefficients (normed space): {results['coefficients_normed_space'].cpu().numpy()}")
            
            # Check reconstruction quality
            orig_norm = torch.norm(results['residual_original']).item()
            print(f"  Original space:")
            print(f"    Residual norm: {orig_norm:.3f}")
            print(f"    Reconstruction error: {results['error_original']:.3f}")
            print(f"    Error %: {100 * results['error_original'] / orig_norm:.1f}%")
            
            normed_norm = torch.norm(results['residual_normed']).item()
            print(f"  Normalized space:")
            print(f"    Residual norm: {normed_norm:.3f}")
            print(f"    Reconstruction error: {results['error_normed']:.3f}")
            print(f"    Error %: {100 * results['error_normed'] / normed_norm:.1f}%")
            
            # Verify reversal works
            reversal_error = torch.norm(
                results['residual_original'] - results['residual_restored']
            ).item()
            print(f"  Layer norm reversal error: {reversal_error:.10f}")


def plot_reconstruction_comparison(model, text, device='cuda'):
    """
    Compare reconstruction quality in different spaces across layers.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    n_layers = model.config.n_layer
    n_positions = input_ids.shape[1]
    
    errors_original = []
    errors_normed = []
    
    for layer_idx in range(n_layers):
        layer_errors_orig = []
        layer_errors_norm = []
        
        for pos_idx in range(n_positions):
            results = decompose_with_reversible_ln(model, input_ids, layer_idx, pos_idx)
            
            # Normalized errors (as % of original norm)
            orig_norm = torch.norm(results['residual_original']).item()
            normed_norm = torch.norm(results['residual_normed']).item()
            
            layer_errors_orig.append(100 * results['error_original'] / orig_norm)
            layer_errors_norm.append(100 * results['error_normed'] / normed_norm)
        
        errors_original.append(np.mean(layer_errors_orig))
        errors_normed.append(np.mean(layer_errors_norm))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_layers), errors_original, label='Original Space', marker='o')
    plt.plot(range(n_layers), errors_normed, label='Normalized Space', marker='s')
    plt.xlabel('Layer')
    plt.ylabel('Reconstruction Error (%)')
    plt.title('Reconstruction Quality: Original vs Normalized Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return errors_original, errors_normed


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox"
    
    # Test perfect reversal
    test_reversible_layer_norm(model, text, device)
    
    # Analyze decomposition in both spaces
    analyze_decomposition_spaces(model, text, device)
    
    # Plot comparison
    print("\n" + "=" * 80)
    print("Plotting reconstruction quality comparison...")
    errors_orig, errors_norm = plot_reconstruction_comparison(model, text, device)
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("1. With stored statistics, layer norm is perfectly reversible")
    print("2. Decomposition can be done in either space:")
    print("   - Normalized space: Where LogitLens operates")
    print("   - Original space: Where residuals naturally live")
    print("3. Reconstruction quality may differ between spaces")
    print(f"   - Avg error in original space: {np.mean(errors_orig):.1f}%")
    print(f"   - Avg error in normalized space: {np.mean(errors_norm):.1f}%")


if __name__ == "__main__":
    main()