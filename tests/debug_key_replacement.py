import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def simple_attention_test():
    """Simple test to verify attention computation is working correctly"""
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "The cat sat on"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Input IDs: {input_ids}")
    
    layer_idx = 5
    head_idx = 1
    query_position = len(tokens) - 1  # Last position
    
    print(f"\nTesting layer {layer_idx}, head {head_idx}, query position {query_position}")
    
    # Get original attention pattern first
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        original_attention = outputs.attentions[layer_idx][0, head_idx, query_position, :]
        print(f"Original attention (from HF): {original_attention.cpu().numpy()}")
        print(f"Original attention sum: {original_attention.sum().item():.6f}")
        
    # Now let's manually compute attention to verify our implementation
    attn_layer = model.transformer.h[layer_idx].attn
    
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        print(f"Hidden states shape before layer {layer_idx}: {hidden_states.shape}")
        
        # Apply layer norm before attention (this is what GPT-2 does!)
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute Q, K, V from the NORMED hidden states
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        print(f"Sequence length: {seq_len}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Num heads: {num_heads}")
        print(f"Head dim: {head_dim}")
        
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        print(f"Q shape after reshape: {q.shape}")  # Should be [batch, num_heads, seq_len, head_dim]
        print(f"K shape after reshape: {k.shape}")
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Extract attention pattern for specific head and position
        manual_attention = attention_weights[0, head_idx, query_position, :]
        
        print(f"Manual attention: {manual_attention.cpu().numpy()}")
        print(f"Manual attention sum: {manual_attention.sum().item():.6f}")
        
        # Compare
        diff = torch.abs(original_attention - manual_attention)
        print(f"Max difference: {diff.max().item():.8f}")
        print(f"Mean difference: {diff.mean().item():.8f}")
        
        if diff.max().item() < 1e-5:
            print("✓ Manual attention computation matches HuggingFace!")
        else:
            print("✗ Manual attention computation differs from HuggingFace")
            
        # Now test simple key replacement
        print(f"\n--- Testing simple key replacement ---")
        
        # Replace all keys with the first key
        original_key = k[0, head_idx, 0, :].clone()  # First position's key
        print(f"Original first key norm: {torch.norm(original_key).item():.4f}")
        
        # Make all keys identical to first key
        k_modified = k.clone()
        for pos in range(seq_len):
            k_modified[0, head_idx, pos, :] = original_key
            
        # Recompute attention with modified keys
        scores_modified = torch.matmul(q, k_modified.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_modified = F.softmax(scores_modified, dim=-1)
        modified_attention = attention_modified[0, head_idx, query_position, :]
        
        print(f"Modified attention (all keys = first key): {modified_attention.cpu().numpy()}")
        print(f"Modified attention sum: {modified_attention.sum().item():.6f}")
        
        # This should be uniform since all keys are identical
        expected_uniform = 1.0 / seq_len
        print(f"Expected uniform value: {expected_uniform:.6f}")
        
        uniform_diff = torch.abs(modified_attention - expected_uniform)
        print(f"Max deviation from uniform: {uniform_diff.max().item():.8f}")
        
        if uniform_diff.max().item() < 1e-5:
            print("✓ Uniform attention achieved with identical keys!")
        else:
            print("✗ Attention not uniform with identical keys - something is wrong")

if __name__ == "__main__":
    simple_attention_test()