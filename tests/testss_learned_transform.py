import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

def get_token_through_first_layer(model, token_id, position_idx, seq_length, device='cuda'):
    """
    Pass a single token through the first transformer layer at a specific position.
    
    Args:
        token_id: The token to transform
        position_idx: Position where the token should be placed
        seq_length: Total sequence length (for positional encoding)
    
    Returns:
        The output of the first layer for that token
    """
    # Create dummy input with just our token at the specified position
    # Use padding token (50256) for other positions
    pad_token_id = 50256
    dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=device)
    dummy_input[0, position_idx] = token_id
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(dummy_input)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        
        # Initial hidden state (what goes into layer 0)
        hidden_states = inputs_embeds + position_embeds
        
        # Pass through first transformer block only
        first_block = model.transformer.h[0]
        
        # Create attention mask (attend only to the token position)
        attention_mask = torch.zeros((1, seq_length), device=device)
        attention_mask[0, position_idx] = 1.0
        
        # Extended attention mask for the transformer block
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through first layer
        outputs = first_block(
            hidden_states,
            attention_mask=extended_attention_mask,
        )
        
        first_layer_output = outputs[0]
        
        return first_layer_output[0, position_idx, :]

def train_learned_transformation(model, train_texts, layers_to_train=[1, 2, 3, 4, 5], k=3, epochs=100, lr=0.001, device='cpu'):
    """Train the learned transformation on multiple texts and layers."""
    
    def simple_tokenize(text):
        # Very basic tokenization for common phrases
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
        elif "raining" in text:
            return torch.tensor([[632, 373, 26742]], device=device)
        elif "dog" in text and "Ben" in text:
            # "Ben saw a dog"
            return torch.tensor([[11696, 2497, 257, 3290]], device=device)
        else:
            # Default fallback
            return torch.tensor([[464, 3797, 3332]], device=device)
    
    # Collect training data
    training_data = []
    print(f"Collecting training data from {len(train_texts)} texts and {len(layers_to_train)} layers...")
    
    with torch.no_grad():
        for text in train_texts:
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
    
    # Initialize and train transformation
    transform_model = LinearTransformation(embed_dim=768).to(device)
    optimizer = optim.Adam(transform_model.parameters(), lr=lr)
    
    # Convert training data to tensors
    all_raw_embeddings = torch.stack([ex['raw_embeddings'] for ex in training_data])
    all_target_residuals = torch.stack([ex['target_residual'] for ex in training_data])
    
    n_examples = all_raw_embeddings.shape[0]
    
    print(f"Training linear transformation on {n_examples} examples...")
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Transform embeddings
        transformed_embeddings = transform_model(all_raw_embeddings)
        
        total_loss = 0
        for i in range(n_examples):
            embeddings = transformed_embeddings[i]
            target = all_target_residuals[i]
            
            # Solve least squares
            A = embeddings.T
            b = target.unsqueeze(1)
            coeffs = torch.linalg.pinv(A) @ b
            reconstruction = A @ coeffs
            
            loss = F.mse_loss(reconstruction.squeeze(), target)
            total_loss += loss
        
        avg_loss = total_loss / n_examples
        losses.append(avg_loss.item())
        
        avg_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss.item():.6f}")
    
    print(f"Training completed. Final loss: {losses[-1]:.6f}")
    return transform_model, losses

def get_residual_and_decompose(model, input_ids, layer_idx, position_idx, transformation_method='learned', transform_model=None):
    """
    Get residual stream and decompose into top 3 tokens + dark matter.
    
    Args:
        transformation_method: 'raw', 'first_layer', or 'learned'
        transform_model: The trained LinearTransformation model (required if method='learned')
    """
    device = input_ids.device
    seq_length = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # hidden_states includes embeddings as layer 0, so layer_idx+1
        if layer_idx == -1:
            # Initial embeddings
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            # Before layer layer_idx (so index is layer_idx, which is after layer_idx-1)
            residual_at_pos = outputs.hidden_states[layer_idx][0, position_idx, :]
    
    # Apply layer normalization for LogitLens (all layers need ln_f for vocab projection)
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    # Get top 3
    top3_values, top3_indices = torch.topk(logits[0], k=3)
    
    # Get embeddings based on transformation method
    if transformation_method == 'raw':
        # Original raw embeddings
        top3_embeddings = model.transformer.wte(top3_indices)
        
    elif transformation_method == 'first_layer':
        # First-layer transformed embeddings (original method)
        top3_embeddings = []
        for token_id in top3_indices:
            transformed = get_token_through_first_layer(
                model, token_id.item(), position_idx, seq_length, device
            )
            top3_embeddings.append(transformed)
        top3_embeddings = torch.stack(top3_embeddings)
        
    elif transformation_method == 'learned':
        # Learned transformation
        if transform_model is None:
            raise ValueError("transform_model required for learned transformation method")
        raw_embeddings = model.transformer.wte(top3_indices)
        top3_embeddings = transform_model(raw_embeddings.unsqueeze(0))[0]  # Remove batch dim
        
    else:
        raise ValueError(f"Unknown transformation method: {transformation_method}")
    
    # Use least squares to find coefficients for the three embeddings
    # Solve: residual_at_pos = coeff1 * embedding1 + coeff2 * embedding2 + coeff3 * embedding3
    A = top3_embeddings.T  # Shape: [hidden_dim, 3]
    b = residual_at_pos    # Shape: [hidden_dim]
    
    # Solve least squares: A @ coeffs = b
    coeffs, residuals, rank, s = torch.linalg.lstsq(A, b)
    
    # Reconstruct using the coefficients
    reconstruction = torch.matmul(A, coeffs)
    dark_matter = residual_at_pos - reconstruction
    
    # Compute cosine similarities with intermediate representation
    intermediate_norm = F.normalize(residual_at_pos.unsqueeze(0), p=2, dim=1)
    similarities = {}
    for i, embedding in enumerate(top3_embeddings):
        token_norm = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
        similarity = torch.cosine_similarity(intermediate_norm, token_norm, dim=1)
        similarities[f'word{i+1}'] = similarity.item()
    
    # Store coefficients for analysis
    coefficients = {f'coeff{i+1}': coeffs[i].item() for i in range(3)}
    
    decomposed = {
        'original': residual_at_pos,
        'word1': coeffs[0] * top3_embeddings[0],
        'word2': coeffs[1] * top3_embeddings[1], 
        'word3': coeffs[2] * top3_embeddings[2],
        'dark_matter': dark_matter
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    # Calculate reconstruction error
    error = torch.norm(residual_at_pos - reconstruction).item()
    error_pct = 100 * error / torch.norm(residual_at_pos).item()
    
    return decomposed, token_names, similarities, top3_values, coefficients, error_pct

def compare_transformation_methods(model, transform_model, input_ids, layer_idx, position_idx):
    """Compare all three transformation methods on the same residual."""
    
    print(f"\nComparing transformation methods for layer {layer_idx}, position {position_idx}:")
    print("-" * 60)
    
    methods = ['raw', 'first_layer', 'learned']
    results = {}
    
    for method in methods:
        decomposed, token_names, similarities, top3_values, coefficients, error_pct = get_residual_and_decompose(
            model, input_ids, layer_idx, position_idx, method, transform_model
        )
        results[method] = {
            'error_pct': error_pct,
            'token_names': token_names,
            'coefficients': coefficients,
            'similarities': similarities
        }
        
        print(f"\n{method.upper()} METHOD:")
        print(f"  Tokens: {token_names}")
        print(f"  Reconstruction error: {error_pct:.1f}%")
        print(f"  Coefficients: {[f'{coeff:.3f}' for coeff in coefficients.values()]}")
        print(f"  Cosine similarities: {[f'{sim:.3f}' for sim in similarities.values()]}")
    
    # Show improvements
    raw_error = results['raw']['error_pct']
    first_error = results['first_layer']['error_pct']
    learned_error = results['learned']['error_pct']
    
    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"  First-layer vs Raw: {raw_error - first_error:+.1f} percentage points")
    print(f"  Learned vs Raw: {raw_error - learned_error:+.1f} percentage points")
    print(f"  Learned vs First-layer: {first_error - learned_error:+.1f} percentage points")
    
    return results

def compute_attention_to_decomposed_keys(
    model, 
    input_ids, 
    layer_idx, 
    head_idx,
    query_position,
    key_position,
    transformation_method='learned',
    transform_model=None
):
    """Compute how query at one position attends to decomposed keys at another position."""
    
    attn_layer = model.transformer.h[layer_idx].attn
    
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Apply layer norm before attention
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Get Q from query position
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        query_vector = q[0, head_idx, query_position, :]
        
        # Get decomposed representations at key position
        decomposed, token_names, similarities, _, coefficients, error_pct = get_residual_and_decompose(
            model, input_ids, layer_idx-1, key_position, transformation_method, transform_model
        )
        
        # Compute key vectors for each decomposed representation
        attention_scores = {}
        
        for rep_name, rep_vector in decomposed.items():
            # Apply layer norm to decomposed vector
            rep_normed = model.transformer.h[layer_idx].ln_1(
                rep_vector.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project to key space
            rep_kv = attn_layer.c_attn(rep_normed.unsqueeze(0).unsqueeze(0))
            rep_k = rep_kv[:, :, hidden_dim:2*hidden_dim]
            rep_k = rep_k.view(1, 1, num_heads, head_dim)[0, 0, head_idx, :]
            
            # Compute attention score (before softmax)
            score = torch.dot(query_vector, rep_k) / (head_dim ** 0.5)
            attention_scores[rep_name] = score.item()
    
    return attention_scores, token_names, similarities, coefficients, error_pct

def analyze_decomposed_key_attention_with_learned_transform(
    model, tokenizer, text, transform_model, 
    layer_idx=5, head_idx=0, transformation_method='learned'
):
    """Analyze how queries attend to decomposed keys using learned transformation."""
    
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Layer {layer_idx}, Head {head_idx}")
    print(f"Transformation method: {transformation_method}")
    
    # Pick query position (last token)
    query_position = len(tokens) - 1
    print(f"\nQuery position: {query_position} ('{tokens[query_position]}')")
    
    # Analyze attention to each key position
    all_attention_scores = []
    all_token_names = []
    all_error_pcts = []
    
    for key_pos in range(len(tokens)):
        scores, token_names, similarities, coefficients, error_pct = compute_attention_to_decomposed_keys(
            model, input_ids, layer_idx, head_idx, query_position, key_pos, 
            transformation_method, transform_model
        )
        all_attention_scores.append(scores)
        all_token_names.append(token_names)
        all_error_pcts.append(error_pct)
        
        print(f"\nKey position {key_pos} ('{tokens[key_pos]}'):")
        print(f"  Reconstruction error: {error_pct:.1f}%")
        print(f"  Top 3 projections: {token_names}")
        print(f"  Least squares coefficients:")
        for i, (key, coeff) in enumerate(coefficients.items()):
            token_name = token_names[i]
            print(f"    {key} ('{token_name}'): {coeff:.4f}")
        print(f"  Cosine similarities:")
        for i, (key, sim) in enumerate(similarities.items()):
            token_name = token_names[i]
            print(f"    {key} ('{token_name}'): {sim:.4f}")
        print(f"  Attention scores:")
        for rep_name, score in scores.items():
            if rep_name.startswith('word'):
                idx = int(rep_name[-1]) - 1
                print(f"    {rep_name} ('{token_names[idx]}'): {score:.4f}")
            else:
                print(f"    {rep_name}: {score:.4f}")
    
    # Show average reconstruction error
    avg_error = np.mean(all_error_pcts)
    print(f"\nAverage reconstruction error across positions: {avg_error:.1f}%")
    
    # Visualize
    visualize_decomposed_key_attention_with_errors(
        all_attention_scores, tokens, all_token_names, all_error_pcts,
        layer_idx, head_idx, query_position, transformation_method
    )
    
    return all_attention_scores, all_token_names, all_error_pcts

def visualize_decomposed_key_attention_with_errors(
    all_attention_scores, tokens, all_token_names, all_error_pcts,
    layer_idx, head_idx, query_position, transformation_method
):
    """Visualize attention from query to decomposed keys with error information."""
    
    n_positions = len(tokens)
    rep_types = ['original', 'word1', 'word2', 'word3', 'dark_matter']
    
    # Create matrix of attention scores
    attention_matrix = np.zeros((len(rep_types), n_positions))
    for pos, scores in enumerate(all_attention_scores):
        for i, rep_type in enumerate(rep_types):
            attention_matrix[i, pos] = scores[rep_type]
    
    # Use generic labels for y-axis
    y_labels = ['original', 'word1', 'word2', 'word3', 'dark_matter']
    
    # Apply softmax across all decomposed keys for each position
    attention_softmax = np.zeros_like(attention_matrix)
    for pos in range(n_positions):
        scores = attention_matrix[:, pos]
        exp_scores = np.exp(scores - scores.max())
        attention_softmax[:, pos] = exp_scores / exp_scores.sum()
    
    # Create visualization with error information
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Raw attention scores
    im1 = ax1.imshow(attention_matrix, aspect='auto', cmap='coolwarm')
    ax1.set_yticks(range(len(rep_types)))
    ax1.set_yticklabels(y_labels)
    ax1.set_xticks(range(n_positions))
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.set_title(f'Raw Attention Scores from Query "{tokens[query_position]}" to Decomposed Keys\n'
                  f'Method: {transformation_method}')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Representation Type')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations for all values
    for i in range(len(rep_types)):
        for j in range(n_positions):
            val = attention_matrix[i, j]
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color='white' if val < 0 else 'black', fontsize=8)
    
    # Plot 2: Softmax normalized 
    im2 = ax2.imshow(attention_softmax, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_yticks(range(len(rep_types)))
    ax2.set_yticklabels(y_labels)
    ax2.set_xticks(range(n_positions))
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.set_title('Softmax Normalized Attention (Relative Within Each Position)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Representation Type')
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations
    for i in range(len(rep_types)):
        for j in range(n_positions):
            val = attention_softmax[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if val > 0.5 else 'black', fontsize=8)
    
    # Plot 3: Reconstruction errors by position
    bars = ax3.bar(range(n_positions), all_error_pcts, alpha=0.7, color='steelblue')
    ax3.set_xticks(range(n_positions))
    ax3.set_xticklabels(tokens, rotation=45, ha='right')
    ax3.set_ylabel('Reconstruction Error (%)')
    ax3.set_title(f'Reconstruction Error by Position (Method: {transformation_method})')
    ax3.grid(True, alpha=0.3)
    
    # Add error values on bars
    for i, (bar, error) in enumerate(zip(bars, all_error_pcts)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{error:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add average error line
    avg_error = np.mean(all_error_pcts)
    ax3.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8, 
               label=f'Average: {avg_error:.1f}%')
    ax3.legend()
    
    plt.suptitle(
        f'Attention to Decomposed Keys with {transformation_method.title()} Transform\n'
        f'Layer {layer_idx}, Head {head_idx}, Query Position {query_position}',
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    print("=" * 70)
    print("TESTSS WITH LEARNED TRANSFORMATION")
    print("=" * 70)
    
    # Step 1: Train learned transformation
    train_texts = [
        "The cat sat",
        "She was walking", 
        "John likes pizza",
        "The book contains",
        "We are going",
        "It was raining"
    ]
    
    print("Training learned transformation...")
    transform_model, losses = train_learned_transformation(
        model, train_texts, layers_to_train=[1, 2, 3, 4, 5], 
        k=3, epochs=50, lr=0.001, device=device
    )
    
    # Step 2: Test text
    test_text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(test_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nTest text: '{test_text}'")
    print(f"Tokens: {tokens}")
    
    # Step 3: Compare transformation methods on different layers/positions
    print("\n" + "=" * 70)
    print("COMPARING TRANSFORMATION METHODS")
    print("=" * 70)
    
    for layer_idx in [2, 5]:
        for position_idx in [0, 2, -1]:  # First, middle, last
            if position_idx == -1:
                position_idx = len(tokens) - 1
            compare_transformation_methods(model, transform_model, input_ids, layer_idx, position_idx)
    
    # Step 4: Analyze attention with learned transformation
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS WITH LEARNED TRANSFORMATION")
    print("=" * 70)
    
    # Analyze different layers and heads
    for layer_idx in [2, 5]:
        for head_idx in [0, 3]:
            analyze_decomposed_key_attention_with_learned_transform(
                model, tokenizer, test_text, transform_model,
                layer_idx, head_idx, transformation_method='learned'
            )
    
    # Step 5: Compare with original first-layer method
    print("\n" + "=" * 70)
    print("COMPARISON WITH FIRST-LAYER TRANSFORMATION")
    print("=" * 70)
    
    analyze_decomposed_key_attention_with_learned_transform(
        model, tokenizer, test_text, transform_model,
        layer_idx=5, head_idx=0, transformation_method='first_layer'
    )

if __name__ == "__main__":
    main()