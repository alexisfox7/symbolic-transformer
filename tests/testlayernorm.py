import torch
from transformers import GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt

def analyze_ln_f_parameters(model):
    """
    Analyze the parameters of GPT-2's final layer norm (ln_f).
    """
    ln_f = model.transformer.ln_f
    
    print("=" * 80)
    print("GPT-2 ln_f (Final Layer Norm) Parameters")
    print("=" * 80)
    
    # Basic info
    print(f"\nModule type: {type(ln_f)}")
    print(f"Epsilon (for numerical stability): {ln_f.eps}")
    print(f"Elementwise affine: {ln_f.elementwise_affine}")
    print(f"Normalized shape: {ln_f.normalized_shape}")
    
    # Learned parameters
    print(f"\n--- LEARNED PARAMETERS ---")
    print(f"Weight shape: {ln_f.weight.shape}")
    print(f"Bias shape: {ln_f.bias.shape}")
    
    # Statistics of learned parameters
    weight_np = ln_f.weight.detach().cpu().numpy()
    bias_np = ln_f.bias.detach().cpu().numpy()
    
    print(f"\nWeight statistics:")
    print(f"  Mean: {weight_np.mean():.6f}")
    print(f"  Std: {weight_np.std():.6f}")
    print(f"  Min: {weight_np.min():.6f}")
    print(f"  Max: {weight_np.max():.6f}")
    print(f"  Median: {np.median(weight_np):.6f}")
    
    print(f"\nBias statistics:")
    print(f"  Mean: {bias_np.mean():.6f}")
    print(f"  Std: {bias_np.std():.6f}")
    print(f"  Min: {bias_np.min():.6f}")
    print(f"  Max: {bias_np.max():.6f}")
    print(f"  Median: {np.median(bias_np):.6f}")
    
    # Show first few values
    print(f"\nFirst 10 weight values: {weight_np[:10]}")
    print(f"First 10 bias values: {bias_np[:10]}")
    
    return weight_np, bias_np


def visualize_ln_f_parameters(weight, bias):
    """
    Visualize the distribution of ln_f parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Weight distribution
    ax = axes[0, 0]
    ax.hist(weight, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(weight.mean(), color='red', linestyle='--', label=f'Mean: {weight.mean():.3f}')
    ax.axvline(1.0, color='green', linestyle='--', label='1.0 (no scaling)')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.set_title('ln_f Weight Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bias distribution
    ax = axes[0, 1]
    ax.hist(bias, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(bias.mean(), color='red', linestyle='--', label=f'Mean: {bias.mean():.3f}')
    ax.axvline(0.0, color='green', linestyle='--', label='0.0 (no shift)')
    ax.set_xlabel('Bias Value')
    ax.set_ylabel('Count')
    ax.set_title('ln_f Bias Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Weight values by dimension
    ax = axes[1, 0]
    ax.plot(weight, alpha=0.7)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Weight Value')
    ax.set_title('ln_f Weights by Dimension')
    ax.grid(True, alpha=0.3)
    
    # Bias values by dimension
    ax = axes[1, 1]
    ax.plot(bias, alpha=0.7, color='orange')
    ax.axhline(0.0, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Bias Value')
    ax.set_title('ln_f Biases by Dimension')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_ln_f_computation(model, text='The quick brown fox'):
    """
    Show exactly how ln_f transforms a vector.
    """
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATING ln_f COMPUTATION")
    print("=" * 80)
    
    with torch.no_grad():
        # Get a residual vector from layer 5
        outputs = model(input_ids, output_hidden_states=True)
        residual = outputs.hidden_states[5][0, 2, :]  # Layer 5, position 2
        
        print(f"\nInput residual shape: {residual.shape}")
        print(f"Input residual norm: {torch.norm(residual).item():.3f}")
        print(f"Input residual mean: {residual.mean().item():.6f}")
        print(f"Input residual std: {residual.std().item():.6f}")
        
        # Manual computation
        ln_f = model.transformer.ln_f
        eps = ln_f.eps
        
        # Step 1: Compute mean and variance
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(dim=-1, keepdim=True, unbiased=False)
        
        print(f"\nStep 1 - Statistics:")
        print(f"  Mean: {mean.item():.6f}")
        print(f"  Variance: {variance.item():.6f}")
        print(f"  Std: {torch.sqrt(variance).item():.6f}")
        
        # Step 2: Normalize
        x_normalized = (residual - mean) / torch.sqrt(variance + eps)
        
        print(f"\nStep 2 - After normalization:")
        print(f"  Normalized mean: {x_normalized.mean().item():.6f} (should be ~0)")
        print(f"  Normalized std: {x_normalized.std().item():.6f} (should be ~1)")
        print(f"  Normalized norm: {torch.norm(x_normalized).item():.3f}")
        
        # Step 3: Apply learned affine transformation
        output_manual = ln_f.weight * x_normalized + ln_f.bias
        
        print(f"\nStep 3 - After affine transform:")
        print(f"  Output mean: {output_manual.mean().item():.6f}")
        print(f"  Output std: {output_manual.std().item():.6f}")
        print(f"  Output norm: {torch.norm(output_manual).item():.3f}")
        
        # Compare with PyTorch's implementation
        output_pytorch = ln_f(residual)
        
        diff = torch.norm(output_manual - output_pytorch).item()
        print(f"\nVerification:")
        print(f"  Difference from PyTorch implementation: {diff:.10f}")
        print(f"  Match: {diff < 1e-5}")
        
        # Show the transformation formula
        print("\n" + "=" * 80)
        print("FORMULA:")
        print("=" * 80)
        print("ln_f(x) = weight * ((x - mean(x)) / sqrt(var(x) + eps)) + bias")
        print("\nWhere:")
        print("  - mean and var are computed over the hidden dimension")
        print("  - weight and bias are learned parameters (shape: [768])")
        print("  - eps = 1e-5 (for numerical stability)")


def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Analyze parameters
    weight, bias = analyze_ln_f_parameters(model)
    
    # Visualize
    visualize_ln_f_parameters(weight, bias)
    
    # Demonstrate computation
    demonstrate_ln_f_computation(model)
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. ln_f has 768 learned weights and 768 learned biases")
    print("2. Weights are centered around 1.0 (minimal scaling)")
    print("3. Biases are centered around 0.0 (minimal shifting)")
    print("4. The normalization (mean=0, std=1) happens BEFORE applying weights/biases")
    print("5. This is different from batch norm - statistics are computed per-token, not per-batch")


if __name__ == "__main__":
    main()