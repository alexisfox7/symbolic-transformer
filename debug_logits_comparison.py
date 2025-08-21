#!/usr/bin/env python3
"""
Debug script to compare logits between testss.py and logitlens_gpt2.py approaches
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx):
    """testss.py approach - but only process up to layer_idx like logitlens"""
    device = input_ids.device
    attention_mask = torch.ones_like(input_ids).bool()
    
    with torch.no_grad():
        # Get embeddings  
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device))
        hidden_states = inputs_embeds + position_embeds
        
        # Process through layers up to layer_idx
        for i in range(layer_idx + 1):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
        
        residual_at_pos = hidden_states[0, position_idx, :]
    
    # Project residual directly to vocabulary (standard LogitLens)
    # Note: GPT2 has weight tying, so this is the correct approach
    logits = model.lm_head(residual_at_pos.unsqueeze(0))
    
    return logits[0], residual_at_pos

def logitlens_gpt2_approach(model, input_ids, layer_idx, position_idx):
    """logitlens_gpt2.py approach - manual layer processing"""
    device = input_ids.device
    attention_mask = torch.ones_like(input_ids).bool()
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device))
        hidden_states = inputs_embeds + position_embeds
        
        # Process through layers up to layer_idx
        for i in range(layer_idx + 1):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
        
        # Get residual at position
        residual_at_pos = hidden_states[0, position_idx, :]
        
        # Apply language modeling head directly to residual (standard LogitLens)
        logits = model.lm_head(hidden_states)
        
        return logits[0, position_idx, :], residual_at_pos

def compare_approaches():
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    # Test text
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    # Test parameters
    layer_idx = 5  # Middle layer
    position_idx = input_ids.shape[1] - 1  # Last position
    
    print(f"Comparing approaches for layer {layer_idx}, position {position_idx}")
    print(f"Text: '{text}'")
    print(f"Input shape: {input_ids.shape}")
    
    # Method 1: testss.py approach
    logits1, residual1 = get_residual_and_decompose_simple(model, input_ids, layer_idx, position_idx)
    top_indices1 = torch.topk(logits1, k=5).indices
    top_tokens1 = [tokenizer.decode([idx.item()]) for idx in top_indices1]
    
    print(f"\nMethod 1 (testss.py - output_hidden_states):")
    print(f"Residual shape: {residual1.shape}")
    print(f"Logits shape: {logits1.shape}")
    print(f"Top 5 tokens: {top_tokens1}")
    print(f"Top 5 logits: {logits1[top_indices1].tolist()}")
    
    # Method 2: logitlens_gpt2.py approach  
    logits2, residual2 = logitlens_gpt2_approach(model, input_ids, layer_idx, position_idx)
    top_indices2 = torch.topk(logits2, k=5).indices
    top_tokens2 = [tokenizer.decode([idx.item()]) for idx in top_indices2]
    
    print(f"\nMethod 2 (logitlens_gpt2.py - manual processing):")
    print(f"Residual shape: {residual2.shape}")
    print(f"Logits shape: {logits2.shape}")
    print(f"Top 5 tokens: {top_tokens2}")
    print(f"Top 5 logits: {logits2[top_indices2].tolist()}")
    
    # Compare residuals
    residual_diff = torch.abs(residual1 - residual2).max()
    logits_diff = torch.abs(logits1 - logits2).max()
    
    print(f"\nDifferences:")
    print(f"Max residual difference: {residual_diff.item():.10f}")
    print(f"Max logits difference: {logits_diff.item():.10f}")
    print(f"Residuals are close: {torch.allclose(residual1, residual2, rtol=1e-5, atol=1e-8)}")
    print(f"Logits are close: {torch.allclose(logits1, logits2, rtol=1e-5, atol=1e-8)}")
    
    # Compare with actual model output
    print(f"\nComparing with actual model output:")
    with torch.no_grad():
        actual_outputs = model(input_ids)
        actual_logits = actual_outputs.logits[0, position_idx, :]
        actual_top_indices = torch.topk(actual_logits, k=5).indices
        actual_top_tokens = [tokenizer.decode([idx.item()]) for idx in actual_top_indices]
        
    print(f"Actual model top 5: {actual_top_tokens}")
    print(f"Actual logits: {actual_logits[actual_top_indices].tolist()}")
    print(f"Method 1 vs Actual max diff: {torch.abs(logits1 - actual_logits).max().item():.10f}")
    print(f"Method 2 vs Actual max diff: {torch.abs(logits2 - actual_logits).max().item():.10f}")

if __name__ == "__main__":
    compare_approaches()