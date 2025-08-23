# rvq_decomp_layers.py
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx):
    """
    Simpler version using HuggingFace's output_hidden_states.
    """
    device = input_ids.device
    
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
    top3_indices = torch.topk(logits[0], k=3).indices
    top3_embeddings = model.transformer.wte(top3_indices)
    
    decomposed = {
        'original': residual_at_pos,
        'word1': top3_embeddings[0],
        'word2': top3_embeddings[1],
        'word3': top3_embeddings[2],
        'dark_matter': residual_at_pos - top3_embeddings.sum(dim=0)
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_names = [tokenizer.decode([idx.item()]) for idx in top3_indices]
    
    return decomposed, token_names

def analyze_rvq_decomp_all_layers(model, tokenizer, text):
    """
    Analyze RVQ decomposition for the last position across all layers.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"\nAnalyzing: '{text}'")
    print(f"Tokens: {tokens}")
    
    # Analyze last position
    query_position = len(tokens) - 1
    print(f"\nRVQ Decomposition at position {query_position} ('{tokens[query_position]}'):")
    print("="*80)
    
    # Get decompositions for all layers
    all_layer_decomps = {}
    
    # Include embedding layer
    for layer_idx in range(-1, model.config.n_layer):
        decomposed, token_names = get_residual_and_decompose_simple(
            model, input_ids, layer_idx, query_position
        )
        
        all_layer_decomps[layer_idx] = {
            'decomposed': decomposed,
            'token_names': token_names
        }
        
        layer_name = "Embeddings" if layer_idx == -1 else f"Layer {layer_idx}"
        print(f"\n{layer_name}:")
        for i, name in enumerate(token_names):
            print(f"  {i+1}. '{name}'")
    
    return all_layer_decomps, tokens, query_position

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    # Test text
    text = "The quick brown fox"
    
    # Analyze RVQ decomposition across all layers
    all_decomps, tokens, pos = analyze_rvq_decomp_all_layers(model, tokenizer, text)
    
    return all_decomps

if __name__ == "__main__":
    results = main()