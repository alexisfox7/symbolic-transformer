#!/usr/bin/env python3
"""
Detailed debug script to find other issues between approaches
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def testss_approach(model, input_ids, layer_idx, position_idx):
    """testss.py approach using output_hidden_states"""
    device = input_ids.device
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # hidden_states includes embeddings as layer 0, so layer_idx+1
        if layer_idx == -1:
            # Initial embeddings
            residual_at_pos = outputs.hidden_states[0][0, position_idx, :]
        else:
            # After layer layer_idx (so index is layer_idx + 1)
            residual_at_pos = outputs.hidden_states[layer_idx + 1][0, position_idx, :]
    
    # Apply final LN and project (standard LogitLens)
    residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
    logits = model.lm_head(residual_normed)
    
    return logits[0], residual_at_pos

def logitlens_approach(model, input_ids, layer_idx, position_idx):
    """logitlens_gpt2.py approach - manual processing"""
    device = input_ids.device
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device))
        hidden_states = inputs_embeds + position_embeds
        
        # Process through layers up to layer_idx
        for i in range(layer_idx + 1):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Get residual at position
        residual_at_pos = hidden_states[0, position_idx, :]
        
        # Apply layer norm and project
        normalized_states = model.transformer.ln_f(hidden_states)
        logits = model.lm_head(normalized_states)
        
        return logits[0, position_idx, :], residual_at_pos

def debug_step_by_step():
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    # Test text
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    print(f"Input text: '{text}'")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")
    
    # Test at different layers
    position_idx = input_ids.shape[1] - 1
    
    for layer_idx in [0, 2, 5, 11]:  # Different layers
        print(f"\n{'='*60}")
        print(f"Testing layer {layer_idx}, position {position_idx}")
        
        # Method 1: testss.py approach
        logits1, residual1 = testss_approach(model, input_ids, layer_idx, position_idx)
        top_indices1 = torch.topk(logits1, k=3).indices
        top_tokens1 = [tokenizer.decode([idx.item()]) for idx in top_indices1]
        
        # Method 2: logitlens_gpt2.py approach  
        logits2, residual2 = logitlens_approach(model, input_ids, layer_idx, position_idx)
        top_indices2 = torch.topk(logits2, k=3).indices
        top_tokens2 = [tokenizer.decode([idx.item()]) for idx in top_indices2]
        
        print(f"Method 1 (testss.py): {top_tokens1}")
        print(f"Method 2 (logitlens): {top_tokens2}")
        
        # Detailed comparison
        residual_diff = torch.abs(residual1 - residual2).max()
        logits_diff = torch.abs(logits1 - logits2).max()
        residual_close = torch.allclose(residual1, residual2, rtol=1e-5, atol=1e-8)
        logits_close = torch.allclose(logits1, logits2, rtol=1e-5, atol=1e-8)
        
        print(f"Max residual diff: {residual_diff.item():.10f}")
        print(f"Max logits diff: {logits_diff.item():.10f}")
        print(f"Residuals close: {residual_close}")
        print(f"Logits close: {logits_close}")
        
        if not residual_close or not logits_close:
            print("⚠️  MISMATCH DETECTED!")
            
            # Check if it's due to attention mask differences
            print("Investigating attention mask effects...")
            
            # Test without attention mask in logitlens approach
            device = input_ids.device
            with torch.no_grad():
                inputs_embeds = model.transformer.wte(input_ids)
                position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device))
                hidden_states_no_mask = inputs_embeds + position_embeds
                
                for i in range(layer_idx + 1):
                    block = model.transformer.h[i]
                    # Try without attention mask
                    hidden_states_no_mask = block(hidden_states_no_mask)[0]
                
                residual_no_mask = hidden_states_no_mask[0, position_idx, :]
                
            residual_mask_diff = torch.abs(residual1 - residual_no_mask).max()
            print(f"Residual diff (testss vs no-mask manual): {residual_mask_diff.item():.10f}")
            
            if torch.allclose(residual1, residual_no_mask, rtol=1e-5, atol=1e-8):
                print("✅ Issue is attention mask in logitlens approach!")
            else:
                print("❌ Issue is NOT attention mask...")
                
                # Let's check the actual hidden states from output_hidden_states
                with torch.no_grad():
                    outputs = model(input_ids, output_hidden_states=True)
                    true_hidden_at_layer = outputs.hidden_states[layer_idx + 1][0, position_idx, :]
                
                print(f"True hidden state from model: {true_hidden_at_layer[:5]}")
                print(f"Manual computation: {residual2[:5]}")
                print(f"Diff with true: {torch.abs(residual2 - true_hidden_at_layer).max().item():.10f}")

if __name__ == "__main__":
    debug_step_by_step()