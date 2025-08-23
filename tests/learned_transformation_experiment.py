import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

class LinearTransformation(nn.Module):
    """Learnable linear transformation for token embeddings."""
    def __init__(self, embed_dim=768):
        super().__init__()
        # Initialize as identity matrix to start close to raw embeddings
        self.transform = nn.Linear(embed_dim, embed_dim, bias=True)
        # Initialize weights close to identity
        nn.init.eye_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, k, embed_dim] or [k, embed_dim]
        Returns:
            transformed embeddings of same shape
        """
        return self.transform(embeddings)


def get_reconstruction_data(model, texts, layers_to_train=[1, 2, 3, 4, 5], k=3, device='cpu'):
    """Collect reconstruction training data from multiple texts and layers."""
    
    # Use simple tokenization to avoid network issues
    def simple_tokenize(text):
        # Very basic tokenization for common phrases
        if "cat sat" in text:
            return torch.tensor([[464, 3797, 3332]], device=device)  # "The cat sat"
        elif "walking" in text:
            return torch.tensor([[3347, 373, 6155]], device=device)  # "She was walking"
        elif "pizza" in text:
            return torch.tensor([[7554, 7832, 14281]], device=device)  # "John likes pizza"
        elif "book" in text:
            return torch.tensor([[464, 1492, 4909]], device=device)  # "The book contains"
        elif "going" in text:
            return torch.tensor([[775, 389, 1016]], device=device)  # "We are going"
        else:
            return torch.tensor([[632, 373, 26742]], device=device)  # "It was raining"
    
    training_data = []
    
    print(f"Collecting training data from {len(texts)} texts and {len(layers_to_train)} layers...")
    
    with torch.no_grad():
        for text in texts:
            input_ids = simple_tokenize(text)
            outputs = model(input_ids, output_hidden_states=True)
            
            for layer_idx in layers_to_train:
                # Get residual at this layer (only first position for simplicity)
                residual = outputs.hidden_states[layer_idx][0, 0, :]  # [embed_dim]
                
                # Get top-k tokens via LogitLens
                residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
                logits = model.lm_head(residual_normed)[0]
                topk_values, topk_indices = torch.topk(logits, k=k)
                
                # Get raw embeddings for these tokens
                raw_embeddings = model.transformer.wte(topk_indices)  # [k, embed_dim]
                
                # Store training example
                training_data.append({
                    'raw_embeddings': raw_embeddings.clone(),
                    'target_residual': residual.clone(),
                    'layer': layer_idx,
                    'text': text
                })
    
    print(f"Collected {len(training_data)} training examples")
    return training_data


def train_linear_transformation(training_data, embed_dim=768, epochs=100, lr=0.001, device='cpu'):
    """Train linear transformation to minimize reconstruction loss."""
    
    # Initialize model
    transform_model = LinearTransformation(embed_dim).to(device)
    optimizer = optim.Adam(transform_model.parameters(), lr=lr)
    
    # Convert training data to tensors
    raw_embeddings_list = []
    target_residuals_list = []
    
    for example in training_data:
        raw_embeddings_list.append(example['raw_embeddings'])
        target_residuals_list.append(example['target_residual'])
    
    # Stack all examples
    all_raw_embeddings = torch.stack(raw_embeddings_list)  # [n_examples, k, embed_dim]
    all_target_residuals = torch.stack(target_residuals_list)  # [n_examples, embed_dim]
    
    n_examples = all_raw_embeddings.shape[0]
    k = all_raw_embeddings.shape[1]
    
    print(f"Training linear transformation on {n_examples} examples...")
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Transform embeddings
        transformed_embeddings = transform_model(all_raw_embeddings)  # [n_examples, k, embed_dim]
        
        total_loss = 0
        
        # Compute reconstruction loss for each example
        for i in range(n_examples):
            # Get transformed embeddings for this example
            embeddings = transformed_embeddings[i]  # [k, embed_dim]
            target = all_target_residuals[i]  # [embed_dim]
            
            # Solve least squares: embeddings.T @ coeffs = target
            A = embeddings.T  # [embed_dim, k]
            b = target.unsqueeze(1)  # [embed_dim, 1]
            
            # Solve using pseudoinverse to avoid singular matrices
            coeffs = torch.linalg.pinv(A) @ b  # [k, 1]
            reconstruction = A @ coeffs  # [embed_dim, 1]
            
            # L2 reconstruction loss
            loss = F.mse_loss(reconstruction.squeeze(), target)
            total_loss += loss
        
        avg_loss = total_loss / n_examples
        losses.append(avg_loss.item())
        
        # Backward pass
        avg_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss.item():.6f}")
    
    print(f"Training completed. Final loss: {losses[-1]:.6f}")
    
    return transform_model, losses


def evaluate_transformations(model, transform_model, test_texts, layers_to_test=[1, 2, 3, 4, 5], k=3, device='cpu'):
    """Compare raw, first-layer, and learned transformations."""
    
    def simple_tokenize(text):
        if "cat sat" in text:
            return torch.tensor([[464, 3797, 3332]], device=device)
        elif "walking" in text:
            return torch.tensor([[3347, 373, 6155]], device=device)
        elif "pizza" in text:
            return torch.tensor([[7554, 7832, 14281]], device=device)
        elif "book" in text:
            return torch.tensor([[464, 1492, 4909]], device=device)
        elif "going" in text:
            return torch.tensor([[775, 389, 1016]], device=device)
        else:
            return torch.tensor([[632, 373, 26742]], device=device)
    
    def get_token_through_first_layer(token_id, position_idx, seq_length):
        """First-layer transformation for comparison."""
        pad_token_id = 50256
        dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=device)
        dummy_input[0, position_idx] = token_id
        
        with torch.no_grad():
            inputs_embeds = model.transformer.wte(dummy_input)
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
            position_embeds = model.transformer.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            
            first_block = model.transformer.h[0]
            attention_mask = torch.zeros((1, seq_length), device=device)
            attention_mask[0, position_idx] = 1.0
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            outputs = first_block(hidden_states, attention_mask=extended_attention_mask)
            return outputs[0][0, position_idx, :]
    
    results = {
        'layers': layers_to_test,
        'raw_errors': [],
        'first_layer_errors': [],
        'learned_errors': []
    }
    
    print(f"Evaluating on {len(test_texts)} test texts...")
    
    with torch.no_grad():
        for layer_idx in layers_to_test:
            layer_raw_errors = []
            layer_first_errors = []
            layer_learned_errors = []
            
            for text in test_texts:
                input_ids = simple_tokenize(text)
                seq_length = input_ids.shape[1]
                outputs = model(input_ids, output_hidden_states=True)
                
                # Get target residual
                residual = outputs.hidden_states[layer_idx][0, 0, :]
                
                # Get top-k tokens
                residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
                logits = model.lm_head(residual_normed)[0]
                topk_values, topk_indices = torch.topk(logits, k=k)
                
                # Method 1: Raw embeddings
                raw_embeddings = model.transformer.wte(topk_indices)
                
                # Method 2: First-layer transformed embeddings
                first_layer_embeddings = []
                for token_id in topk_indices:
                    transformed = get_token_through_first_layer(token_id, 0, seq_length)
                    first_layer_embeddings.append(transformed)
                first_layer_embeddings = torch.stack(first_layer_embeddings)
                
                # Method 3: Learned transformation
                learned_embeddings = transform_model(raw_embeddings.unsqueeze(0))[0]  # Remove batch dim
                
                # Compute reconstruction errors
                def compute_reconstruction_error(embeddings, target):
                    A = embeddings.T
                    b = target.unsqueeze(1)
                    coeffs = torch.linalg.pinv(A) @ b
                    reconstruction = A @ coeffs
                    return torch.norm(target - reconstruction.squeeze()).item()
                
                raw_error = compute_reconstruction_error(raw_embeddings, residual)
                first_error = compute_reconstruction_error(first_layer_embeddings, residual)
                learned_error = compute_reconstruction_error(learned_embeddings, residual)
                
                # Convert to percentage error
                residual_norm = torch.norm(residual).item()
                raw_error_pct = 100 * raw_error / residual_norm
                first_error_pct = 100 * first_error / residual_norm
                learned_error_pct = 100 * learned_error / residual_norm
                
                layer_raw_errors.append(raw_error_pct)
                layer_first_errors.append(first_error_pct)
                layer_learned_errors.append(learned_error_pct)
            
            # Average across texts for this layer
            results['raw_errors'].append(np.mean(layer_raw_errors))
            results['first_layer_errors'].append(np.mean(layer_first_errors))
            results['learned_errors'].append(np.mean(layer_learned_errors))
            
            print(f"Layer {layer_idx}: Raw={np.mean(layer_raw_errors):.1f}%, "
                  f"First-layer={np.mean(layer_first_errors):.1f}%, "
                  f"Learned={np.mean(layer_learned_errors):.1f}%")
    
    return results


def plot_comparison(results, training_losses, save_path='learned_transformation_comparison.png'):
    """Plot comparison of all three methods."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Reconstruction errors
    layers = results['layers']
    ax1.plot(layers, results['raw_errors'], 'o-', linewidth=2.5, markersize=8, 
             label='Raw Embeddings', color='#e74c3c')
    ax1.plot(layers, results['first_layer_errors'], 's-', linewidth=2.5, markersize=8, 
             label='First-Layer Transform', color='#3498db')
    ax1.plot(layers, results['learned_errors'], '^-', linewidth=2.5, markersize=8, 
             label='Learned Transform', color='#2ecc71')
    
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('Reconstruction Error (%)', fontsize=14)
    ax1.set_title('Reconstruction Error Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # Plot 2: Training loss
    ax2.plot(training_losses, linewidth=2, color='#2ecc71')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Training Loss', fontsize=14)
    ax2.set_title('Linear Transformation Training Loss', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    # Training texts
    train_texts = [
        "The cat sat",
        "She was walking", 
        "John likes pizza",
        "The book contains"
    ]
    
    # Test texts (different from training)
    test_texts = [
        "We are going",
        "It was raining"
    ]
    
    layers_to_use = [1, 2, 3, 4, 5]
    k = 3  # Number of top tokens
    
    print("=" * 70)
    print("LEARNED LINEAR TRANSFORMATION EXPERIMENT")
    print("=" * 70)
    
    # Step 1: Collect training data
    training_data = get_reconstruction_data(
        model, train_texts, layers_to_use, k=k, device=device
    )
    
    # Step 2: Train linear transformation
    transform_model, training_losses = train_linear_transformation(
        training_data, embed_dim=768, epochs=100, lr=0.001, device=device
    )
    
    # Step 3: Evaluate all methods
    results = evaluate_transformations(
        model, transform_model, test_texts, layers_to_use, k=k, device=device
    )
    
    # Step 4: Create comparison plot
    save_path = plot_comparison(results, training_losses)
    print(f"\nComparison plot saved as: {save_path}")
    
    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    avg_raw = np.mean(results['raw_errors'])
    avg_first = np.mean(results['first_layer_errors'])
    avg_learned = np.mean(results['learned_errors'])
    
    print(f"Average reconstruction error across layers {layers_to_use}:")
    print(f"  Raw embeddings:           {avg_raw:.1f}%")
    print(f"  First-layer transform:    {avg_first:.1f}%")
    print(f"  Learned transform:        {avg_learned:.1f}%")
    
    print(f"\nImprovements over raw embeddings:")
    print(f"  First-layer improvement:  {avg_raw - avg_first:+.1f} percentage points")
    print(f"  Learned improvement:      {avg_raw - avg_learned:+.1f} percentage points")
    
    print(f"\nLearned vs First-layer:")
    if avg_learned < avg_first:
        print(f"  Learned is better by:     {avg_first - avg_learned:+.1f} percentage points")
    else:
        print(f"  First-layer is better by: {avg_learned - avg_first:+.1f} percentage points")
    
    # Best method
    best_errors = [avg_raw, avg_first, avg_learned]
    best_names = ['Raw', 'First-layer', 'Learned']
    best_idx = np.argmin(best_errors)
    
    print(f"\nBest method: {best_names[best_idx]} ({best_errors[best_idx]:.1f}% error)")


if __name__ == "__main__":
    main()