#!/usr/bin/env python3
"""
Debug the layer norm difference
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_layernorm_difference():
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    layer_idx = 5
    position_idx = input_ids.shape[1] - 1
    
    with torch.no_grad():
        # Method 1: Natural path through transformer block
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Now let's see what happens in the transformer block for layer 5
        block_layer5 = model.transformer.h[layer_idx]
        
        # The natural path: ln_1 -> attention -> residual connection -> ln_2 -> mlp -> residual
        ln1_output_natural = block_layer5.ln_1(hidden_states)
        print(f"Natural ln_1 output at pos {position_idx}: {ln1_output_natural[0, position_idx, :5]}")
        
        # Method 2: Manual application to extracted residual
        residual_at_pos = hidden_states[0, position_idx, :]
        ln1_output_manual = block_layer5.ln_1(residual_at_pos.unsqueeze(0).unsqueeze(0))
        print(f"Manual ln_1 output: {ln1_output_manual[0, 0, :5]}")
        
        print(f"Layer norm outputs are same: {torch.allclose(ln1_output_natural[0, position_idx, :], ln1_output_manual[0, 0, :])}")
        
        # If layer norms are the same, let's check the c_attn computation
        attn_layer = block_layer5.attn
        
        natural_qkv = attn_layer.c_attn(ln1_output_natural)
        manual_qkv = attn_layer.c_attn(ln1_output_manual)
        
        hidden_dim = hidden_states.shape[-1]
        natural_q = natural_qkv[:, :, :hidden_dim]
        manual_q = manual_qkv[:, :, :hidden_dim]
        
        print(f"Natural c_attn query at pos {position_idx}: {natural_q[0, position_idx, :5]}")
        print(f"Manual c_attn query: {manual_q[0, 0, :5]}")
        print(f"c_attn outputs are same: {torch.allclose(natural_q[0, position_idx, :], manual_q[0, 0, :])}")

if __name__ == "__main__":
    debug_layernorm_difference()