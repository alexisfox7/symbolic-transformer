#!/usr/bin/env python3
"""
Test the fixed original behavior in testss.py
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys
sys.path.append('/Users/alexisfox/st')

# Import the fixed functions
from testss import get_residual_and_decompose_simple, compute_attention_with_modified_query, analyze_decomposed_attention

def test_fixed_original():
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    layer_idx = 5
    head_idx = 0
    position_idx = input_ids.shape[1] - 1
    
    print(f"Testing fixed implementation:")
    print(f"Layer {layer_idx}, Head {head_idx}, Position {position_idx}")
    
    # Get true attention pattern
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        true_attention = outputs.attentions[layer_idx][0, head_idx, position_idx, :]
        
    print(f"True attention pattern: {true_attention[:8].tolist()}")
    print(f"True max attention: {true_attention.max().item():.4f}")
    
    # Get decomposed representations with fixed function
    decomposed, token_names = get_residual_and_decompose_simple(
        model, input_ids, layer_idx, position_idx
    )
    
    print(f"Vocab tokens: {token_names}")
    
    # Test "original" with fixed function
    original_residual = decomposed['original']
    print(f"Original residual: {original_residual[:5].tolist()}")
    
    # Compute attention with fixed function
    fixed_attention = compute_attention_with_modified_query(
        model, input_ids, layer_idx, head_idx, position_idx, original_residual
    )
    
    print(f"Fixed 'original' attention: {fixed_attention[:8].tolist()}")  
    print(f"Fixed max attention: {fixed_attention.max().item():.4f}")
    
    # Check if it's still the problematic [1.0, 0.0, 0.0...] pattern
    is_sharp = (fixed_attention[0] > 0.9 and fixed_attention[1:].max() < 0.1)
    print(f"Still showing sharp [1.0, 0, 0...] pattern: {is_sharp}")
    
    # Compare with true pattern
    diff = torch.abs(true_attention - fixed_attention).max()
    print(f"Max difference from true: {diff.item():.6f}")
    print(f"Patterns close: {torch.allclose(true_attention, fixed_attention, rtol=1e-2)}")

if __name__ == "__main__":
    test_fixed_original()