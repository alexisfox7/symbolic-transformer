#!/usr/bin/env python3
"""
Test the current testss.py behavior
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import the function from testss.py
import sys
sys.path.append('/Users/alexisfox/st')
from testss import get_residual_and_decompose_simple

def test_current_behavior():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    position_idx = input_ids.shape[1] - 1
    
    print("Testing current testss.py behavior:")
    
    for layer_idx in [5, 11]:
        print(f"\n=== Layer {layer_idx} ===")
        
        # Current testss.py approach
        decomposed, token_names = get_residual_and_decompose_simple(
            model, input_ids, layer_idx, position_idx
        )
        print(f"testss.py top tokens: {token_names}")
        
        # What it should be (manual approach)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            residual_at_pos = outputs.hidden_states[layer_idx + 1][0, position_idx, :]
            
            if layer_idx == 11:
                # Layer 11 is already normalized in output_hidden_states
                logits_correct = model.lm_head(residual_at_pos.unsqueeze(0))
            else:
                # Other layers need layer norm
                residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
                logits_correct = model.lm_head(residual_normed)
            
            top3_indices = torch.topk(logits_correct[0], k=3).indices
            token_names_correct = [tokenizer.decode([idx.item()]) for idx in top3_indices]
        
        print(f"Correct approach: {token_names_correct}")
        print(f"Match: {token_names == token_names_correct}")

if __name__ == "__main__":
    test_current_behavior()