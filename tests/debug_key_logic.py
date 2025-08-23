import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_key_replacement_logic():
    """Test the exact key replacement logic with simple examples"""
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "The cat sat"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    layer_idx = 5
    head_idx = 1
    query_position = 2  # "sat" position
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Testing layer {layer_idx}, head {head_idx}, query at position {query_position}")
    
    # Get the attention layer
    attn_layer = model.transformer.h[layer_idx].attn
    
    with torch.no_grad():
        # Get normal attention first
        outputs = model(input_ids, output_attentions=True)
        original_attention = outputs.attentions[layer_idx][0, head_idx, query_position, :]
        print(f"Original attention: {original_attention.cpu().numpy()}")
        
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Apply layer norm before attention
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Compute Q, K, V
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Verify manual computation matches
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        manual_attention = F.softmax(scores, dim=-1)[0, head_idx, query_position, :]
        
        diff = torch.abs(original_attention - manual_attention).max()
        print(f"Manual vs HF attention max diff: {diff.item():.8f}")
        
        if diff < 1e-6:
            print("✓ Manual attention matches")
        else:
            print("✗ Manual attention differs")
            return
        
        # Now test key replacement
        print(f"\n--- Testing key replacement ---")
        
        # Create a simple test: replace all keys with position 0's key
        original_key_0 = k[0, head_idx, 0, :].clone()
        print(f"Original key 0 values: {original_key_0[:5].cpu().numpy()}")
        
        # Method 1: Direct replacement
        k_test1 = k.clone()
        for pos in range(seq_len):
            k_test1[0, head_idx, pos, :] = original_key_0
            
        scores_test1 = torch.matmul(q, k_test1.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_test1 = F.softmax(scores_test1, dim=-1)[0, head_idx, query_position, :]
        print(f"Test 1 attention (all keys = key[0]): {attention_test1.cpu().numpy()}")
        
        # This should be uniform
        expected_uniform = 1.0 / seq_len
        uniform_error = torch.abs(attention_test1 - expected_uniform).max()
        print(f"Uniform error: {uniform_error.item():.8f}")
        
        if uniform_error < 1e-6:
            print("✓ Uniform attention with identical keys")
        else:
            print("✗ Non-uniform attention with identical keys")
        
        # Now test with different vectors for different positions
        print(f"\n--- Testing with different vectors ---")
        
        # Create some simple different key vectors
        k_test2 = k.clone()
        base_vector = original_key_0
        
        for pos in range(seq_len):
            # Create slightly different vectors
            if pos == 0:
                k_test2[0, head_idx, pos, :] = base_vector
            elif pos == 1:
                k_test2[0, head_idx, pos, :] = base_vector * 1.5  # Scale up
            elif pos == 2:
                k_test2[0, head_idx, pos, :] = base_vector * 0.5  # Scale down
            
        scores_test2 = torch.matmul(q, k_test2.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_test2 = F.softmax(scores_test2, dim=-1)[0, head_idx, query_position, :]
        print(f"Test 2 attention (scaled keys): {attention_test2.cpu().numpy()}")
        
        # This should NOT be uniform
        max_attention = attention_test2.max()
        min_attention = attention_test2.min()
        print(f"Attention range: {min_attention.item():.6f} to {max_attention.item():.6f}")
        
        if (max_attention - min_attention) > 0.01:
            print("✓ Non-uniform attention with different keys")
        else:
            print("✗ Still uniform attention with different keys")

if __name__ == "__main__":
    debug_key_replacement_logic()