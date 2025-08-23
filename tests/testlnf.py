import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt

def analyze_basis_change_relationship(model, text, device='cuda'):
    """
    Test if first layer transformation is related to inverse of ln_f.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_length = len(tokens)
    
    print("=" * 80)
    print("TESTING: Is Layer 0 an 'inverse' of ln_f?")
    print("=" * 80)
    
    # Pick a test token
    test_token = " fox"
    token_id = tokenizer.encode(test_token, add_special_tokens=False)[0]
    position = 2
    
    print(f"\nTest token: '{test_token}' at position {position}")
    print("-" * 60)
    
    with torch.no_grad():
        # 1. Get raw embedding
        raw_embed = model.transformer.wte(torch.tensor([token_id], device=device))[0]
        pos_embed = model.transformer.wpe(torch.tensor([position], device=device))[0]
        initial_input = raw_embed + pos_embed
        
        # 2. Pass through first layer
        dummy_input = torch.full((1, seq_length), 50256, dtype=torch.long, device=device)
        dummy_input[0, position] = token_id
        
        inputs_embeds = model.transformer.wte(dummy_input)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Through first layer
        first_block = model.transformer.h[0]
        attention_mask = torch.ones((1, seq_length), device=device)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        layer0_output = first_block(hidden_states, attention_mask=extended_attention_mask)[0]
        layer0_token = layer0_output[0, position, :]
        
        # 3. Apply ln_f to both
        initial_normed = model.transformer.ln_f(initial_input.unsqueeze(0))[0]
        layer0_normed = model.transformer.ln_f(layer0_token.unsqueeze(0))[0]
        
        # 4. Analyze the transformation
        print("\nNorm Analysis:")
        print(f"  Raw embedding norm:          {torch.norm(raw_embed).item():.3f}")
        print(f"  Initial (embed+pos) norm:    {torch.norm(initial_input).item():.3f}")
        print(f"  After layer 0 norm:          {torch.norm(layer0_token).item():.3f}")
        print(f"  Initial after ln_f norm:     {torch.norm(initial_normed).item():.3f}")
        print(f"  Layer0 after ln_f norm:      {torch.norm(layer0_normed).item():.3f}")
        
        print("\nStatistics before ln_f:")
        print(f"  Initial mean: {initial_input.mean().item():.6f}, std: {initial_input.std().item():.6f}")
        print(f"  Layer0 mean:  {layer0_token.mean().item():.6f}, std: {layer0_token.std().item():.6f}")
        
        print("\nStatistics after ln_f:")
        print(f"  Initial→ln_f mean: {initial_normed.mean().item():.6f}, std: {initial_normed.std().item():.6f}")
        print(f"  Layer0→ln_f mean:  {layer0_normed.mean().item():.6f}, std: {layer0_normed.std().item():.6f}")
        
        # 5. Test if layer0 "pre-normalizes" in a way that undoes ln_f
        # Hypothesis: layer0 applies a transformation similar to inverse ln_f
        
        # What would "ideal" pre-normalization look like?
        # If we want output of ln_f to be some target, what should input be?
        
        # Let's see if layer0 output is more "ln_f-ready" than raw embedding
        # by checking how much ln_f changes each
        
        change_initial = torch.norm(initial_normed - initial_input).item()
        change_layer0 = torch.norm(layer0_normed - layer0_token).item()
        
        print(f"\nChange caused by ln_f:")
        print(f"  Initial→ln_f change: {change_initial:.3f} ({100*change_initial/torch.norm(initial_input).item():.1f}%)")
        print(f"  Layer0→ln_f change:  {change_layer0:.3f} ({100*change_layer0/torch.norm(layer0_token).item():.1f}%)")
        
        return {
            'raw_embed': raw_embed,
            'initial_input': initial_input,
            'layer0_output': layer0_token,
            'initial_normed': initial_normed,
            'layer0_normed': layer0_normed
        }


def test_reconstruction_with_unnormalized(model, text, device='cuda'):
    """
    Test if using un-normalized layer 0 outputs works better for later layers.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print("\n" + "=" * 80)
    print("RECONSTRUCTION TEST: Does skipping ln_f help when using Layer 0 outputs?")
    print("=" * 80)
    
    layer_idx = 5
    pos_idx = 2
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        target_residual = outputs.hidden_states[layer_idx][0, pos_idx, :]
        
        # Get top 3 tokens via LogitLens
        residual_normed = model.transformer.ln_f(target_residual.unsqueeze(0))
        logits = model.lm_head(residual_normed)[0]
        top3_indices = torch.topk(logits, k=3).indices
        
        print(f"\nTarget: Layer {layer_idx}, Position {pos_idx} ('{tokens[pos_idx]}')")
        print(f"Top 3 tokens: {[tokenizer.decode([idx.item()]) for idx in top3_indices]}")
        print(f"Target residual norm: {torch.norm(target_residual).item():.3f}")
        
        # Method 1: Raw embeddings with ln_f normalization (standard)
        raw_embeddings = model.transformer.wte(top3_indices)
        raw_embeddings_normed = model.transformer.ln_f(raw_embeddings)
        
        # Method 2: Layer 0 outputs WITHOUT additional ln_f
        layer0_embeddings = []
        for token_id in top3_indices:
            transformed = get_token_through_first_layer(
                model, token_id, pos_idx, len(tokens), device
            )
            layer0_embeddings.append(transformed)
        layer0_embeddings = torch.stack(layer0_embeddings)
        
        # Method 3: Layer 0 outputs WITH ln_f
        layer0_embeddings_normed = model.transformer.ln_f(layer0_embeddings)
        
        # Compare reconstructions
        methods = [
            ("Raw embeddings", raw_embeddings),
            ("Raw + ln_f", raw_embeddings_normed),
            ("Layer0 outputs", layer0_embeddings),
            ("Layer0 + ln_f", layer0_embeddings_normed)
        ]
        
        print("\nReconstruction Results:")
        print("-" * 60)
        
        for name, embeddings in methods:
            # Least squares reconstruction
            A = embeddings.T
            b = target_residual.unsqueeze(1)
            coeffs = torch.linalg.lstsq(A, b).solution.squeeze()
            reconstruction = (A @ coeffs.unsqueeze(1)).squeeze()
            error = torch.norm(target_residual - reconstruction).item()
            error_pct = 100 * error / torch.norm(target_residual).item()
            
            # Average cosine similarity
            avg_cosine = np.mean([
                F.cosine_similarity(target_residual.unsqueeze(0), 
                                   embeddings[i].unsqueeze(0)).item()
                for i in range(3)
            ])
            
            print(f"\n{name:20s}:")
            print(f"  Coefficients: {coeffs.cpu().numpy()}")
            print(f"  Error: {error:.3f} ({error_pct:.1f}%)")
            print(f"  Avg cosine sim: {avg_cosine:.3f}")
            print(f"  Embedding norms: {[f'{torch.norm(e).item():.1f}' for e in embeddings]}")


def test_inverse_relationship(model, device='cuda'):
    """
    Test if Layer 0 learns to invert the statistics that ln_f will later normalize.
    """
    print("\n" + "=" * 80)
    print("TESTING INVERSE RELATIONSHIP")
    print("=" * 80)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Test multiple tokens
    test_tokens = [" the", " quick", " brown", " fox", " dog", " cat"]
    
    correlations = {
        'norm_growth': [],
        'mean_shift': [],
        'std_change': []
    }
    
    for token_str in test_tokens:
        token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]
        
        with torch.no_grad():
            # Get embedding
            raw_embed = model.transformer.wte(torch.tensor([token_id], device=device))[0]
            
            # Pass through layer 0 at different positions
            positions = [0, 2, 4, 6]
            for pos in positions:
                # Get position embedding
                pos_embed = model.transformer.wpe(torch.tensor([pos], device=device))[0]
                initial = raw_embed + pos_embed
                
                # Through layer 0
                transformed = get_token_through_first_layer(
                    model, token_id, pos, 10, device
                )
                
                # Measure changes
                norm_ratio = torch.norm(transformed).item() / torch.norm(initial).item()
                mean_change = transformed.mean().item() - initial.mean().item()
                std_ratio = transformed.std().item() / initial.std().item()
                
                correlations['norm_growth'].append(norm_ratio)
                correlations['mean_shift'].append(mean_change)
                correlations['std_change'].append(std_ratio)
    
    print("\nStatistics across tokens and positions:")
    print(f"  Norm growth (layer0/initial):     mean={np.mean(correlations['norm_growth']):.3f}, std={np.std(correlations['norm_growth']):.3f}")
    print(f"  Mean shift:                       mean={np.mean(correlations['mean_shift']):.6f}, std={np.std(correlations['mean_shift']):.6f}")
    print(f"  Std change (layer0/initial):      mean={np.mean(correlations['std_change']):.3f}, std={np.std(correlations['std_change']):.3f}")
    
    print("\nInterpretation:")
    if np.mean(correlations['norm_growth']) > 5:
        print("  ✓ Layer 0 dramatically increases norm (preparing for later normalization)")
    elif np.mean(correlations['norm_growth']) > 1.5:
        print("  → Layer 0 moderately increases norm")
    else:
        print("  ✗ Layer 0 doesn't significantly change norm")
    
    return correlations


def get_token_through_first_layer(model, token_id, position_idx, seq_length, device='cuda'):
    """
    Helper function (same as before) to get token through first layer.
    """
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


def visualize_transformation_path(results):
    """
    Visualize how vectors transform through the pipeline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Collect data
    stages = ['Raw Embed', 'Initial\n(+position)', 'After Layer0', 'After ln_f']
    
    # Norms
    norms = [
        torch.norm(results['raw_embed']).item(),
        torch.norm(results['initial_input']).item(),
        torch.norm(results['layer0_output']).item(),
        torch.norm(results['layer0_normed']).item()
    ]
    
    axes[0, 0].bar(stages, norms)
    axes[0, 0].set_ylabel('Norm')
    axes[0, 0].set_title('Norm at Each Stage')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Means
    means = [
        results['raw_embed'].mean().item(),
        results['initial_input'].mean().item(),
        results['layer0_output'].mean().item(),
        results['layer0_normed'].mean().item()
    ]
    
    axes[0, 1].bar(stages, means)
    axes[0, 1].set_ylabel('Mean')
    axes[0, 1].set_title('Mean at Each Stage')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Stds
    stds = [
        results['raw_embed'].std().item(),
        results['initial_input'].std().item(),
        results['layer0_output'].std().item(),
        results['layer0_normed'].std().item()
    ]
    
    axes[1, 0].bar(stages, stds)
    axes[1, 0].set_ylabel('Std Dev')
    axes[1, 0].set_title('Std Dev at Each Stage')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cosine similarities with final
    final = results['layer0_normed']
    cosines = [
        F.cosine_similarity(results['raw_embed'].unsqueeze(0), final.unsqueeze(0)).item(),
        F.cosine_similarity(results['initial_input'].unsqueeze(0), final.unsqueeze(0)).item(),
        F.cosine_similarity(results['layer0_output'].unsqueeze(0), final.unsqueeze(0)).item(),
        1.0  # With itself
    ]
    
    axes[1, 1].bar(stages, cosines)
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Similarity with Final (Layer0→ln_f)')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Transformation Pipeline Analysis')
    plt.tight_layout()
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The quick brown fox"
    
    # Test 1: Analyze the relationship
    results = analyze_basis_change_relationship(model, text, device)
    
    # Test 2: Reconstruction comparison
    test_reconstruction_with_unnormalized(model, text, device)
    
    # Test 3: Statistical relationship
    correlations = test_inverse_relationship(model, device)
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_transformation_path(results)
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS:")
    print("=" * 80)
    print("1. Check if Layer 0 dramatically changes norms (preparing for normalization)")
    print("2. See if Layer 0 outputs work better WITHOUT additional ln_f")
    print("3. Look for inverse relationship: does Layer 0 'pre-scale' for later ln_f?")


if __name__ == "__main__":
    main()