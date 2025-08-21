#!/usr/bin/env python3
"""
Debug the "original" issue in testss.py
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_original_issue():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    layer_idx = 5
    head_idx = 0
    position_idx = input_ids.shape[1] - 1
    
    print(f"Debugging layer {layer_idx}, head {head_idx}, position {position_idx}")
    
    # Get the actual attention pattern from the model
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        true_attention = outputs.attentions[layer_idx][0, head_idx, position_idx, :]
        
    print(f"True attention pattern: {true_attention[:5].tolist()}")
    print(f"True attention sum: {true_attention.sum().item():.6f}")
    
    # Test testss.py approach
    from testss import get_residual_and_decompose_simple, compute_attention_with_modified_query
    
    # Get decomposed representations
    decomposed, token_names = get_residual_and_decompose_simple(
        model, input_ids, layer_idx, position_idx
    )
    
    # Test "original" attention
    original_residual = decomposed['original']
    print(f"Original residual (after layer {layer_idx}): {original_residual[:5].tolist()}")
    
    # Compute attention with "original" 
    testss_attention = compute_attention_with_modified_query(
        model, input_ids, layer_idx, head_idx, position_idx, original_residual
    )
    
    print(f"testss.py 'original' attention: {testss_attention[:5].tolist()}")
    print(f"testss.py attention sum: {testss_attention.sum().item():.6f}")
    
    # Check difference
    diff = torch.abs(true_attention - testss_attention).max()
    print(f"Max difference: {diff.item():.6f}")
    print(f"Patterns match: {torch.allclose(true_attention, testss_attention, rtol=1e-3)}")
    
    # The issue: testss.py uses residual AFTER layer_idx but processes layers 0 to layer_idx-1
    print(f"\n=== The Issue ===")
    print(f"testss.py compute_attention_with_modified_query:")
    print(f"  - Processes layers 0 to {layer_idx-1}")
    print(f"  - Uses residual AFTER layer {layer_idx}")
    print(f"  - This creates inconsistency!")
    
    # What should happen: use residual BEFORE layer_idx (i.e., after layer_idx-1)
    print(f"\n=== Correct Approach ===")
    if layer_idx > 0:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            # Residual before layer_idx (i.e., after layer_idx-1)
            correct_residual = outputs.hidden_states[layer_idx][0, position_idx, :]
            
        print(f"Correct residual (after layer {layer_idx-1}): {correct_residual[:5].tolist()}")
        
        # This should match better (but still won't be perfect due to other issues)
        correct_attention = compute_attention_with_modified_query(
            model, input_ids, layer_idx, head_idx, position_idx, correct_residual
        )
        
        print(f"Corrected attention: {correct_attention[:5].tolist()}")
        correct_diff = torch.abs(true_attention - correct_attention).max()
        print(f"Corrected max difference: {correct_diff.item():.6f}")
        print(f"Improved: {correct_diff < diff}")

if __name__ == "__main__":
    debug_original_issue()