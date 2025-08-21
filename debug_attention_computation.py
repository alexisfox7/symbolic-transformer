#!/usr/bin/env python3
"""
Debug the attention computation step by step
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_attention_step_by_step():
    model = GPT2LMHeadModel.from_pretrained('gpt2', attn_implementation="eager")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "Ben saw a dog. Ben saw a dog. Ben saw a"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    
    layer_idx = 5
    head_idx = 0
    position_idx = input_ids.shape[1] - 1
    
    # Get the residual before layer 5 (what we should be using)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        residual_before_layer5 = outputs.hidden_states[layer_idx][0, position_idx, :]  # After layer 4
        
    print(f"Using residual before layer {layer_idx}: {residual_before_layer5[:5]}")
    
    # Let's manually trace what happens in compute_attention_with_modified_query
    attn_layer = model.transformer.h[layer_idx].attn
    
    with torch.no_grad():
        # Step 1: Get hidden states up to layer 4 (before layer 5)
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):  # 0 to 4
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        print(f"Hidden states after layer {layer_idx-1}: {hidden_states[0, position_idx, :5]}")
        print(f"Matches expected residual: {torch.allclose(hidden_states[0, position_idx, :], residual_before_layer5)}")
        
        # Step 2: Compute original Q, K, V for layer 5
        qkv = attn_layer.c_attn(hidden_states)
        hidden_dim = hidden_states.shape[-1]
        q_orig, k_orig, v_orig = qkv.split(hidden_dim, dim=-1)
        
        # Step 3: Apply layer norm to our residual and compute modified Q, K, V
        ln_layer = model.transformer.h[layer_idx].ln_1
        modified_input_normed = ln_layer(residual_before_layer5.unsqueeze(0).unsqueeze(0))
        modified_qkv = attn_layer.c_attn(modified_input_normed)
        modified_q_flat = modified_qkv[:, :, :hidden_dim]
        
        print(f"Original input to ln_1: {hidden_states[0, position_idx, :5]}")
        print(f"Modified input to ln_1: {residual_before_layer5[:5]}")
        print(f"Should be the same: {torch.allclose(hidden_states[0, position_idx, :], residual_before_layer5)}")
        
        # The issue might be here - we're using the SAME residual but getting different attention
        # This suggests the problem is elsewhere in the computation
        
        # Let's check if the issue is in the query replacement logic
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        k = k_orig.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_orig.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = q_orig.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Check original query at position
        print(f"Original query at pos {position_idx}: {q[0, head_idx, position_idx, :5]}")
        
        # Modified query
        modified_q = modified_q_flat.view(1, 1, num_heads, head_dim).transpose(1, 2)
        print(f"Modified query: {modified_q[0, head_idx, 0, :5]}")
        print(f"Queries are the same: {torch.allclose(q[0, head_idx, position_idx, :], modified_q[0, head_idx, 0, :])}")
        
        # If queries are the same, the attention should be the same
        # If not, there's a bug in the computation

if __name__ == "__main__":
    debug_attention_step_by_step()