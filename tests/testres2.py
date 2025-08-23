import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

class LearnableTokenProjection(nn.Module):
    """
    Learn affine transformations to map token embeddings to residual space.
    """
    def __init__(self, d_model, n_tokens=3):
        super().__init__()
        self.n_tokens = n_tokens
        
        # Learnable parameters for each token
        # Scale and shift: residual ≈ scale * token_embed + shift
        self.scales = nn.Parameter(torch.ones(n_tokens))
        self.shifts = nn.Parameter(torch.zeros(n_tokens, d_model))
        
        # Alternative: single affine transform for all tokens
        self.use_shared_transform = False
        self.shared_scale = nn.Parameter(torch.ones(1))
        self.shared_shift = nn.Parameter(torch.zeros(1, d_model))
        
    def forward(self, token_embeddings):
        """
        Apply learnable transformation to token embeddings.
        
        Args:
            token_embeddings: [n_tokens, d_model]
        Returns:
            transformed: [n_tokens, d_model]
        """
        if self.use_shared_transform:
            return self.shared_scale * token_embeddings + self.shared_shift
        else:
            # Per-token transformation
            scales = self.scales.view(-1, 1)  # [n_tokens, 1]
            return scales * token_embeddings + self.shifts


def learn_decomposition(model, input_ids, layer_idx, position_idx, 
                        n_tokens=3, lr=0.1, n_steps=100, device='cuda'):
    """
    Learn optimal shift/scale to decompose residual as combination of token embeddings.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Get target residual
        if layer_idx == -1:
            target = outputs.hidden_states[0][0, position_idx, :]
        else:
            target = outputs.hidden_states[layer_idx][0, position_idx, :]
        
        # Get top k tokens via LogitLens
        residual_normed = model.transformer.ln_f(target.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        topk_values, topk_indices = torch.topk(logits, k=n_tokens)
        
        # Get base token embeddings
        base_embeddings = model.transformer.wte(topk_indices)  # [k, d_model]
    
    # Initialize learnable projection
    projection = LearnableTokenProjection(model.config.hidden_size, n_tokens).to(device)
    
    # Also learn combination coefficients
    coeffs = nn.Parameter(torch.ones(n_tokens) / n_tokens)
    
    # Optimizer
    params = list(projection.parameters()) + [coeffs]
    optimizer = optim.Adam(params, lr=lr)
    
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Transform token embeddings
        transformed_embeddings = projection(base_embeddings)
        
        # Combine with learnable coefficients
        reconstruction = (transformed_embeddings.T @ coeffs.unsqueeze(1)).squeeze()
        
        # Loss
        loss = torch.norm(target - reconstruction) ** 2
        
        # Regularization to keep transformations reasonable
        reg_loss = 0.01 * (torch.norm(projection.scales - 1) ** 2 + 
                           torch.norm(projection.shifts) ** 2)
        
        total_loss = loss + reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss = {loss.item():.4f}, "
                  f"Scales = {projection.scales.data.cpu().numpy()}, "
                  f"Coeffs = {coeffs.data.cpu().numpy()}")
    
    # Final results
    with torch.no_grad():
        transformed_embeddings = projection(base_embeddings)
        final_reconstruction = (transformed_embeddings.T @ coeffs.unsqueeze(1)).squeeze()
        
        error = torch.norm(target - final_reconstruction).item()
        target_norm = torch.norm(target).item()
        error_percent = 100 * error / target_norm
        
        token_names = [tokenizer.decode([idx.item()]) for idx in topk_indices]
        
        results = {
            'tokens': token_names,
            'coefficients': coeffs.data.cpu().numpy(),
            'scales': projection.scales.data.cpu().numpy(),
            'shift_norms': torch.norm(projection.shifts, dim=1).cpu().numpy(),
            'target_norm': target_norm,
            'reconstruction_error': error,
            'error_percent': error_percent,
            'losses': losses,
            'final_reconstruction': final_reconstruction,
            'target': target
        }
    
    return results, projection


def analyze_learned_decomposition(model, text, device='cuda'):
    """
    Analyze learned decomposition across layers.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    
    results_by_layer = {}
    
    for layer_idx in [0, 3, 6, 9, 11]:
        print(f"\nLayer {layer_idx}:")
        print("-" * 60)
        
        layer_results = []
        
        for pos_idx in range(len(tokens)):
            print(f"\nPosition {pos_idx} ('{tokens[pos_idx]}'):")
            
            # Learn decomposition
            results, projection = learn_decomposition(
                model, input_ids, layer_idx, pos_idx,
                n_tokens=3, lr=0.1, n_steps=100, device=device
            )
            
            print(f"  Top tokens: {results['tokens']}")
            print(f"  Learned coefficients: {results['coefficients']}")
            print(f"  Learned scales: {results['scales']}")
            print(f"  Shift norms: {results['shift_norms']}")
            print(f"  Target norm: {results['target_norm']:.3f}")
            print(f"  Final error: {results['reconstruction_error']:.3f} "
                  f"({results['error_percent']:.1f}%)")
            
            layer_results.append(results)
        
        results_by_layer[layer_idx] = layer_results
    
    return results_by_layer


def compare_methods(model, text, device='cuda'):
    """
    Compare different decomposition methods.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print("\n" + "=" * 80)
    print("COMPARISON: Simple Sum vs Least Squares vs Learned Transform")
    print("=" * 80)
    
    layer_idx = 5
    pos_idx = 1
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        target = outputs.hidden_states[layer_idx][0, pos_idx, :]
        target_norm = torch.norm(target).item()
        
        # Get top 3 tokens
        residual_normed = model.transformer.ln_f(target.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        top3_indices = torch.topk(logits, k=3).indices
        top3_embeddings = model.transformer.wte(top3_indices)
        
        # Method 1: Simple sum
        simple_recon = top3_embeddings.sum(dim=0)
        simple_error = torch.norm(target - simple_recon).item()
        
        # Method 2: Least squares (no transform)
        A = top3_embeddings.T
        b = target.unsqueeze(1)
        lstsq_coeffs = torch.linalg.lstsq(A, b).solution.squeeze()
        lstsq_recon = (A @ lstsq_coeffs.unsqueeze(1)).squeeze()
        lstsq_error = torch.norm(target - lstsq_recon).item()
    
    # Method 3: Learned transform
    results, _ = learn_decomposition(
        model, input_ids, layer_idx, pos_idx,
        n_tokens=3, lr=0.1, n_steps=200, device=device
    )
    
    print(f"\nLayer {layer_idx}, Position {pos_idx}:")
    print(f"Target norm: {target_norm:.3f}")
    print(f"\n1. Simple Sum:")
    print(f"   Error: {simple_error:.3f} ({100*simple_error/target_norm:.1f}%)")
    print(f"\n2. Least Squares (no transform):")
    print(f"   Coefficients: {lstsq_coeffs.cpu().numpy()}")
    print(f"   Error: {lstsq_error:.3f} ({100*lstsq_error/target_norm:.1f}%)")
    print(f"\n3. Learned Transform:")
    print(f"   Coefficients: {results['coefficients']}")
    print(f"   Scales: {results['scales']}")
    print(f"   Error: {results['reconstruction_error']:.3f} ({results['error_percent']:.1f}%)")


def plot_learning_curves(results_by_layer):
    """
    Plot learning curves for different layers.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    layer_indices = [0, 3, 6, 9, 11]
    
    for idx, layer_idx in enumerate(layer_indices):
        ax = axes[idx]
        
        # Plot learning curves for each position
        for pos_idx, results in enumerate(results_by_layer[layer_idx]):
            ax.plot(results['losses'], label=f'Pos {pos_idx}', alpha=0.7)
        
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Summary plot
    ax = axes[5]
    for layer_idx in layer_indices:
        avg_errors = [r['error_percent'] for r in results_by_layer[layer_idx]]
        ax.bar(layer_idx, np.mean(avg_errors), alpha=0.7, label=f'Layer {layer_idx}')
    
    ax.set_title('Average Error by Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Error (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox"
    
    # Compare methods
    compare_methods(model, text, device)
    
    # Full analysis
    print("\n" + "=" * 80)
    print("FULL ANALYSIS WITH LEARNED TRANSFORMS")
    print("=" * 80)
    results_by_layer = analyze_learned_decomposition(model, text, device)
    
    # Plot results
    plot_learning_curves(results_by_layer)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for layer_idx in [0, 3, 6, 9, 11]:
        errors = [r['error_percent'] for r in results_by_layer[layer_idx]]
        print(f"Layer {layer_idx:2d}: Average error = {np.mean(errors):.1f}% "
              f"(range: {min(errors):.1f}% - {max(errors):.1f}%)")


if __name__ == "__main__":
    main()