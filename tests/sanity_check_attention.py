import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_basic_attention_logits(model, input_ids, layer_idx, head_idx, query_position):
    """
    Compute basic attention logits (q*k^T) for a specific layer, head, and query position.
    This is a sanity check without any decomposition.
    """
    
    with torch.no_grad():
        # Get hidden states up to this layer
        hidden_states = model.transformer.wte(input_ids)
        hidden_states = hidden_states + model.transformer.wpe(torch.arange(input_ids.shape[1]))
        
        for i in range(layer_idx):
            block = model.transformer.h[i]
            hidden_states = block(hidden_states)[0]
        
        # Apply layer norm before attention (this is what GPT-2 does!)
        hidden_states_normed = model.transformer.h[layer_idx].ln_1(hidden_states)
        
        # Get attention layer
        attn_layer = model.transformer.h[layer_idx].attn
        
        # Compute Q, K, V from the NORMED hidden states
        qkv = attn_layer.c_attn(hidden_states_normed)
        hidden_dim = hidden_states.shape[-1]
        q, k, v = qkv.split(hidden_dim, dim=-1)
        
        # Reshape for multi-head attention
        batch_size, seq_len = input_ids.shape
        num_heads = attn_layer.num_heads
        head_dim = hidden_dim // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores (logits) before softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Extract logits for specific head and query position
        query_logits = scores[0, head_idx, query_position, :]
        
        return query_logits

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    
    # Test text (same as testss.py)
    text = "The cat sat on the mat"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    tokens = [tokenizer.decode([id]) for id in input_ids[0]]
    
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    
    # Test the same configurations as testss.py
    test_configs = [
        (2, 0), (2, 3), (2, 7),
        (5, 0), (5, 3), (5, 7),
        (8, 0), (8, 3), (8, 7)
    ]
    
    query_position = len(tokens) - 1  # Last token (same as testss.py)
    print(f"\nQuery position: {query_position} ('{tokens[query_position]}')")
    
    for layer_idx, head_idx in test_configs:
        print(f"\n{'='*50}")
        print(f"Layer {layer_idx}, Head {head_idx}")
        
        logits = compute_basic_attention_logits(
            model, input_ids, layer_idx, head_idx, query_position
        )
        
        print(f"Attention logits from '{tokens[query_position]}' to all positions:")
        for pos, logit in enumerate(logits):
            print(f"  Position {pos} ('{tokens[pos]}'): {logit.item():.4f}")

if __name__ == "__main__":
    main()