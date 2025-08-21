#!/usr/bin/env python3
"""
Debug the mystery of layer 11 discrepancy
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_final_layer_mystery():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    position_idx = input_ids.shape[1] - 1
    
    print("=== Investigating the final layer mystery ===")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        model_final_logits = outputs.logits[0, position_idx, :]
        
        # Let's check what hidden_states contains
        print(f"Number of hidden states: {len(outputs.hidden_states)}")
        print(f"Hidden state shapes: {[h.shape for h in outputs.hidden_states[:3]]}...")
        
        # The hidden states should be: [embeddings, layer0_out, layer1_out, ..., layer11_out]
        # So index 12 should be after layer 11
        hidden_after_layer11 = outputs.hidden_states[12][0, position_idx, :]
        
        # Let's also check what happens if we manually apply layer norm to this
        ln_applied = model.transformer.ln_f(hidden_after_layer11.unsqueeze(0).unsqueeze(0))
        manual_logits_from_hidden = model.lm_head(ln_applied)[0, 0, :]
        
    print(f"Model final logits: {model_final_logits[:5]}")
    print(f"Manual logits from hidden_states[12]: {manual_logits_from_hidden[:5]}")
    print(f"Diff: {torch.abs(model_final_logits - manual_logits_from_hidden).max().item():.10f}")
    
    # Now let's manually trace what the model ACTUALLY does internally
    print("\n=== Manual forward pass trace ===")
    
    with torch.no_grad():
        # Start with embeddings
        inputs_embeds = model.transformer.wte(input_ids)
        position_embeds = model.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
        hidden_states = inputs_embeds + position_embeds
        
        print(f"After embeddings: {hidden_states[0, position_idx, :5]}")
        
        # Apply all transformer blocks
        for i in range(12):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
            print(f"After layer {i}: {hidden_states[0, position_idx, :5]}")
        
        print(f"Final hidden before ln_f: {hidden_states[0, position_idx, :5]}")
        
        # Apply final layer norm
        normalized = model.transformer.ln_f(hidden_states)
        print(f"After ln_f: {normalized[0, position_idx, :5]}")
        
        # Apply lm_head
        final_logits = model.lm_head(normalized)
        print(f"Final logits: {final_logits[0, position_idx, :5]}")
        
        print(f"Matches model output: {torch.allclose(final_logits[0, position_idx, :], model_final_logits, rtol=1e-5)}")
    
    # Let's also check if there's something weird about how transformers saves hidden states
    print("\n=== Checking hidden_states indexing ===")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # Check each hidden state
        for i, hidden in enumerate(outputs.hidden_states):
            if i == 0:
                desc = "embeddings"
            elif i <= 12:
                desc = f"after layer {i-1}"
            else:
                desc = f"unknown index {i}"
            print(f"hidden_states[{i}] ({desc}): {hidden[0, position_idx, :5]}")

if __name__ == "__main__":
    debug_final_layer_mystery()