
# def compute_attention_with_modified_query(
#     model, 
#     input_ids, 
#     layer_idx, 
#     head_idx,
#     query_position,
#     modified_query_vector
# ):
#     """
#     Compute attention pattern when replacing query vector at specific position.
#     """
#     # Get the attention layer
#     attn_layer = model.transformer.h[layer_idx].attn
    
#     # First, get the key and value vectors normally
#     with torch.no_grad():
#         # Get hidden states up to this layer
#         hidden_states = model.transformer.wte(input_ids)
#         hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
#         for i in range(layer_idx):
#             block = model.transformer.h[i]
#             hidden_states = block(hidden_states)[0]
        
#         # Compute Q, K, V
#         # Note: GPT2 concatenates QKV, need to split
#         qkv = attn_layer.c_attn(hidden_states)  # [batch, seq_len, 3 * hidden_dim]
#         hidden_dim = hidden_states.shape[-1]
#         q, k, v = qkv.split(hidden_dim, dim=-1)
        
#         # Reshape for multi-head attention
#         batch_size, seq_len = input_ids.shape
#         num_heads = attn_layer.num_heads
#         head_dim = hidden_dim // num_heads
        
#         k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
#         v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
#         q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

#         # Replace query at specific position with modified vector
#         # Apply layer norm before attention (like GPT-2 does)
#         ln_layer = model.transformer.h[layer_idx].ln_1
#         modified_input_normed = ln_layer(modified_query_vector.unsqueeze(0).unsqueeze(0))
        
#         # Project modified vector to query space
#         modified_qkv = attn_layer.c_attn(modified_input_normed)
#         modified_q = modified_qkv[:, :, :hidden_dim]
#         modified_k = modified_qkv[:, :, hidden_dim:2*hidden_dim]

#         modified_q = modified_q.view(1, 1, num_heads, head_dim).transpose(1, 2)
#         modified_k = modified_k.view(1, 1, num_heads, head_dim).transpose(1, 2)

#         # Replace both Q and K at the query position
#         q[:, :, query_position, :] = modified_q[0, :, 0, :]

#         # Compute attention scores
#         scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
#         attention_weights = F.softmax(scores, dim=-1)

#         # Extract attention pattern for specific head and position
#         attention_pattern = attention_weights[0, head_idx, query_position, :]
#         return attention_pattern
    