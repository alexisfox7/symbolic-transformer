import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def debug_decomposed_components():
    """Debug what the decomposed components actually look like"""
    
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    
    text = "The blue fox eats"
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    layer_idx = 5
    query_position = len(tokens) - 1
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Analyzing layer {layer_idx}, position {query_position}")
    
    with torch.no_grad():
        # Get residual stream
        outputs = model(input_ids, output_hidden_states=True)
        residual_at_pos = outputs.hidden_states[layer_idx][0, query_position, :]
        
        print(f"Residual norm: {torch.norm(residual_at_pos).item():.4f}")
        
        # Get top 3 tokens
        residual_normed = model.transformer.ln_f(residual_at_pos.unsqueeze(0))
        logits = model.lm_head(residual_normed)
        top3_values, top3_indices = torch.topk(logits[0], k=3)
        top3_embeddings = model.transformer.wte(top3_indices)
        
        print(f"\nTop 3 tokens:")
        for i, (idx, val, emb) in enumerate(zip(top3_indices, top3_values, top3_embeddings)):
            token_name = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token_name}' (logit: {val.item():.4f}, emb_norm: {torch.norm(emb).item():.4f})")
        
        # Create components
        dark_matter = residual_at_pos - top3_embeddings.sum(dim=0)
        
        components = {
            'original': residual_at_pos,
            'word1': top3_embeddings[0], 
            'word2': top3_embeddings[1],
            'word3': top3_embeddings[2],
            'dark_matter': dark_matter
        }
        
        print(f"\nComponent analysis:")
        for name, comp in components.items():
            norm = torch.norm(comp).item()
            print(f"  {name}: norm={norm:.4f}")
            if name != 'original':
                # Check if it's roughly in same direction as residual
                cos_sim = F.cosine_similarity(comp.unsqueeze(0), residual_at_pos.unsqueeze(0))
                print(f"    cos_sim with residual: {cos_sim.item():.4f}")
        
        # Verify reconstruction
        reconstruction = top3_embeddings.sum(dim=0) + dark_matter
        reconstruction_error = torch.norm(reconstruction - residual_at_pos).item()
        print(f"\nReconstruction error: {reconstruction_error:.8f}")
        
        # Now test what happens when we use these as key vectors
        print(f"\n--- Testing components as key vectors ---")
        
        attn_layer = model.transformer.h[layer_idx].attn
        
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        head_idx = 1
        
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Test each component as keys
        for comp_name, component in components.items():
            if comp_name == 'original':
                continue
                
            print(f"\n  Testing {comp_name} as all keys:")
            
            # Apply layer norm to component (this is key!)
            component_normed = model.transformer.h[layer_idx].ln_1(
                component.unsqueeze(0).unsqueeze(0)
            )[0, 0, :]
            
            # Project to key space
            component_qkv = attn_layer.c_attn(component_normed.unsqueeze(0).unsqueeze(0))
            component_k = component_qkv[:, :, hidden_dim:2*hidden_dim]
            component_k = component_k.view(1, 1, num_heads, head_dim)
            
            # Replace all keys
            k_test = k.clone()
            for pos in range(seq_len):
                k_test[0, head_idx, pos, :] = component_k[0, 0, head_idx, :]
            
            # Compute attention
            scores = torch.matmul(q, k_test.transpose(-2, -1)) / (head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)[0, head_idx, query_position, :]
            
            print(f"    Attention: {attention.cpu().numpy()}")
            print(f"    Max attention: {attention.max().item():.6f}")
            print(f"    Is uniform: {torch.allclose(attention, torch.full_like(attention, 1.0/seq_len), atol=1e-6)}")

if __name__ == "__main__":
    debug_decomposed_components()