#!/usr/bin/env python3
"""
Debug script specifically for layer 11 issue
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def investigate_layer11():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    position_idx = input_ids.shape[1] - 1
    
    print(f"Model has {model.config.n_layer} layers (0-{model.config.n_layer-1})")
    print(f"Testing layer 11 (final layer)")
    
    # Method 1: Get hidden states from model with output_hidden_states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        true_hidden_layer11 = outputs.hidden_states[12][0, position_idx, :]  # layer 11 + 1 = index 12
        final_logits_from_model = outputs.logits[0, position_idx, :]
    
    print(f"True hidden state at layer 11: {true_hidden_layer11[:5]}")
    
    # Method 2: Manual processing through all layers
    device = input_ids.device
    with torch.no_grad():
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device))
        hidden_states = inputs_embeds + position_embeds
        
        print(f"Initial embeddings: {hidden_states[0, position_idx, :5]}")
        
        # Process through each layer and check
        for i in range(12):  # 0 to 11
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
            
            # Get the corresponding hidden state from output_hidden_states for comparison
            true_hidden_at_i = outputs.hidden_states[i+1][0, position_idx, :]
            manual_hidden_at_i = hidden_states[0, position_idx, :]
            
            diff = torch.abs(true_hidden_at_i - manual_hidden_at_i).max()
            print(f"Layer {i}: Max diff = {diff.item():.10f}")
            
            if diff > 1e-5:
                print(f"  ❌ Significant difference at layer {i}!")
                print(f"  True: {true_hidden_at_i[:5]}")
                print(f"  Manual: {manual_hidden_at_i[:5]}")
                break
            elif i == 11:
                print(f"  Layer {i} (final): Manual={manual_hidden_at_i[:5]}")
    
    # Check if the issue is in the layer processing or somewhere else
    manual_final = hidden_states[0, position_idx, :]
    print(f"\nFinal comparison:")
    print(f"True final hidden: {true_hidden_layer11[:5]}")  
    print(f"Manual final: {manual_final[:5]}")
    print(f"Max difference: {torch.abs(true_hidden_layer11 - manual_final).max().item():.10f}")
    
    # Test if the model's final processing matches
    with torch.no_grad():
        # Apply final layer norm and lm_head like the model does
        normalized_manual = model.transformer.ln_f(manual_final.unsqueeze(0).unsqueeze(0))
        manual_logits = model.lm_head(normalized_manual)[0, 0, :]
        
        normalized_true = model.transformer.ln_f(true_hidden_layer11.unsqueeze(0).unsqueeze(0))
        true_logits = model.lm_head(normalized_true)[0, 0, :]
        
    print(f"\nLogits comparison:")
    print(f"Model final logits: {final_logits_from_model[:5]}")
    print(f"True hidden → logits: {true_logits[:5]}")
    print(f"Manual hidden → logits: {manual_logits[:5]}")
    print(f"Model vs True logits diff: {torch.abs(final_logits_from_model - true_logits).max().item():.10f}")
    print(f"Manual vs True logits diff: {torch.abs(manual_logits - true_logits).max().item():.10f}")

if __name__ == "__main__":
    investigate_layer11()