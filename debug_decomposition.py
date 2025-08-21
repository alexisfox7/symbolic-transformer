import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_decomposition_approach():
    """Test different approaches to creating decomposed components"""
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "The cat sat on"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    layer_idx = 5
    query_position = len(tokens) - 1
    
    print(f"Text: {text}")
    print(f"Analyzing layer {layer_idx}, position {query_position}")
    
    with torch.no_grad():
        # Get the residual stream at layer_idx (before that layer processes it)
        outputs = model(input_ids, output_hidden_states=True)
        residual_at_pos = outputs.hidden_states[layer_idx][0, query_position, :]
        
        print(f"Residual stream vector norm: {torch.norm(residual_at_pos).item():.4f}")
        
        # Get top 3 via LogitLens
        residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        top3_values, top3_indices = torch.topk(logits[0], k=3)
        
        print(f"Top 3 tokens:")
        for i, (idx, val) in enumerate(zip(top3_indices, top3_values)):
            token_name = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token_name}' (logit: {val.item():.4f})")
        
        # Test different approaches for creating word components
        print(f"\n--- Testing decomposition approaches ---")
        
        # Approach 1: Raw embeddings (original testss.py approach before my change)
        raw_embeddings = model.transformer.wte(top3_indices)
        
        print(f"Raw embedding approach:")
        for i, emb in enumerate(raw_embeddings):
            token_name = tokenizer.decode([top3_indices[i].item()])
            norm = torch.norm(emb).item()
            print(f"  word{i+1} ('{token_name}') norm: {norm:.4f}")
        
        raw_dark_matter = residual_at_pos - raw_embeddings.sum(dim=0)
        print(f"  dark_matter norm: {torch.norm(raw_dark_matter).item():.4f}")
        
        # Approach 2: Embeddings through first layer (my modification)
        processed_embeddings = []
        for i, embedding in enumerate(raw_embeddings):
            embedding_copy = embedding.clone()
            pos_encoding = model.transformer.wpe(torch.tensor([0], device=embedding.device))
            embedding_with_pos = embedding_copy + pos_encoding[0]
            first_layer_output = model.transformer.h[0](embedding_with_pos.unsqueeze(0).unsqueeze(0))[0]
            processed_embeddings.append(first_layer_output[0, 0, :])
        
        processed_embeddings = torch.stack(processed_embeddings)
        
        print(f"Processed through first layer approach:")
        for i, emb in enumerate(processed_embeddings):
            token_name = tokenizer.decode([top3_indices[i].item()])
            norm = torch.norm(emb).item()
            print(f"  word{i+1} ('{token_name}') norm: {norm:.4f}")
        
        processed_dark_matter = residual_at_pos - processed_embeddings.sum(dim=0)
        print(f"  dark_matter norm: {torch.norm(processed_dark_matter).item():.4f}")
        
        # Check which approach makes more sense
        print(f"\n--- Sanity checks ---")
        
        # The sum of components should be close to original
        raw_reconstruction = raw_embeddings.sum(dim=0) + raw_dark_matter
        raw_error = torch.norm(raw_reconstruction - residual_at_pos).item()
        print(f"Raw approach reconstruction error: {raw_error:.6f}")
        
        processed_reconstruction = processed_embeddings.sum(dim=0) + processed_dark_matter
        processed_error = torch.norm(processed_reconstruction - residual_at_pos).item()
        print(f"Processed approach reconstruction error: {processed_error:.6f}")
        
        # The raw approach should have near-zero error by construction
        if raw_error < 1e-5:
            print("✓ Raw approach has perfect reconstruction (expected)")
        else:
            print("✗ Raw approach reconstruction error too high")
            
        if processed_error > 0.1:
            print("✓ Processed approach has significant reconstruction error (expected)")
        else:
            print("? Processed approach reconstruction error unexpectedly low")

if __name__ == "__main__":
    test_decomposition_approach()