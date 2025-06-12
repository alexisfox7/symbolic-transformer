import torch
import torch.nn.functional as F
import os
import sys

def simple_knowledge_graph_extraction(checkpoint_path: str, text: str):
    """
    Simple knowledge graph extraction - just the basics that work.
    """
    # Add parent directory for imports
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from config import get_preset_config
    from mytokenizers import create_tokenizer
    from modelold import get_model
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint - check what keys are available first
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Use the same logic as check.py to handle different checkpoint formats
    first_key = list(checkpoint.keys())[0] if checkpoint else ""
    
    if first_key.startswith('module.'):
        # Checkpoint IS the model state dict
        print("Checkpoint is model state dict (starts with 'module.')")
        model_state_dict = checkpoint
        
        # Need to create config manually - try to infer size from checkpoint
        # Look at a model weight to infer the model size
        sample_key = next(iter(model_state_dict.keys()))
        sample_weight = model_state_dict[sample_key]
        
        # Try to infer model size from weights
        if 'transformer.wte.weight' in model_state_dict:
            n_embd = model_state_dict['transformer.wte.weight'].shape[1]
            print(f"Inferred n_embd from weights: {n_embd}")
            
            # Map embedding size to preset
            if n_embd == 128:
                preset = 'tiny'
            elif n_embd == 192:
                preset = 'small'  
            elif n_embd == 384:
                preset = 'medium'
            elif n_embd == 768:
                preset = 'large'
            else:
                print(f"Unknown embedding size {n_embd}, using medium as fallback")
                preset = 'medium'
        else:
            print("Could not infer model size, using medium as fallback")
            preset = 'medium'
        
        print(f"Using inferred preset: {preset}")
        config = get_preset_config(preset)
        
        # Remove 'module.' prefix
        fixed_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            fixed_state_dict[new_key] = value
        model_state_dict = fixed_state_dict
        
    else:
        # Standard checkpoint format
        config = checkpoint.get('config')
        if not config:
            print("No config found, trying to infer model size from weights...")
            
            # Try to infer model size from weights
            if 'transformer.wte.weight' in checkpoint:
                n_embd = checkpoint['transformer.wte.weight'].shape[1]
                print(f"Inferred n_embd from weights: {n_embd}")
                
                # Map embedding size to preset
                if n_embd == 128:
                    preset = 'tiny'
                elif n_embd == 192:
                    preset = 'small'  
                elif n_embd == 384:
                    preset = 'medium'
                elif n_embd == 768:
                    preset = 'large'
                else:
                    print(f"Unknown embedding size {n_embd}, using medium as fallback")
                    preset = 'medium'
            else:
                print("Could not infer model size, using medium as fallback")
                preset = 'medium'
            
            print(f"Using inferred preset: {preset}")
            config = get_preset_config(preset)
        
        # Find model state dict
        possible_keys = ['model_state_dict', 'state_dict', 'model']
        model_state_key = None
        for key in possible_keys:
            if key in checkpoint:
                model_state_key = key
                break
        
        if not model_state_key:
            raise ValueError(f"No model state dict found. Available keys: {list(checkpoint.keys())}")
        
        model_state_dict = checkpoint[model_state_key]
    
    print(f"Using config: {config}")
    
    # Create tokenizer first
    tokenizer = create_tokenizer('gpt2')
    
    # Update config with tokenizer info (like your training scripts do)
    config.update_from_tokenizer(tokenizer)
    
    print(f"Config updated with vocab_size: {config.vocab_size}")
    
    # Create model
    model = get_model("Symbolic", config=config)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Create tokenizer
    # tokenizer = create_tokenizer('gpt2')  # Already created above
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt')
    print(f"Input: '{text}'")
    print(f"Tokens: {input_ids.squeeze().tolist()}")
    
    # Forward pass to get outputs
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"Model output shape: {outputs['logits'].shape}")
    
    # Get vocabulary projections from the final vocab grounding layer
    # First get the hidden states from the last transformer layer
    with torch.no_grad():
        # Get token embeddings
        token_embeddings = model.transformer.wte(input_ids)  # (1, seq_len, n_embd)
        
        # Pass through all transformer layers
        hidden_states = token_embeddings
        for layer in model.transformer.h:
            hidden_states = layer(hidden_states)
        
        # Apply final layer norm
        hidden_states = model.transformer.ln_f(hidden_states)
        
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # Now use vocab grounding to get vocabulary weights
        # The VocabularyProjectionFFN has channel_vocab_attentions - let's access those
        print(f"Number of channel vocab attentions: {len(model.vocab_grounding.channel_vocab_attentions)}")
        
        # Get vocabulary weights by channel
        B, T, C = hidden_states.shape
        n_head = model.vocab_grounding.n_head
        head_dim = model.vocab_grounding.head_dim
        
        # Reshape to separate head channels: (B, T, n_head, head_dim)
        hidden_channels = hidden_states.view(B, T, n_head, head_dim)
        
        all_vocab_weights = []
        
        for h in range(n_head):
            # Extract channel h: (B, T, head_dim)
            hidden_h = hidden_channels[:, :, h, :]
            
            # Apply channel-specific FFN transformation
            ffn_h = model.vocab_grounding.channel_ffns[h](hidden_h)  # (B, T, head_dim)
            
            # Compute attention weights over vocabulary for this channel
            vocab_logits_h = model.vocab_grounding.channel_vocab_attentions[h](ffn_h)  # (B, T, vocab_size)
            
            # Apply temperature scaling
            temp_h = torch.clamp(model.vocab_grounding.channel_temperatures[h], min=0.1)
            vocab_weights_h = F.softmax(vocab_logits_h / temp_h, dim=-1)  # (B, T, vocab_size)
            
            all_vocab_weights.append(vocab_weights_h)
        
        # Average across all channels to get final vocabulary weights
        vocab_weights = torch.stack(all_vocab_weights, dim=0).mean(dim=0)  # (B, T, vocab_size)
        vocab_weights = vocab_weights.squeeze(0)  # (T, vocab_size)
        
        print(f"Vocabulary weights shape: {vocab_weights.shape}")
        
        print(f"\nVocabulary composition for each token:")
        print("="*50)
        
        # Show top vocabulary tokens for each position
        for pos in range(vocab_weights.shape[0]):
            original_token = input_ids[0, pos].item()
            original_text = tokenizer.decode([original_token])
            
            # Get top 5 vocabulary tokens for this position
            top_k = torch.topk(vocab_weights[pos], k=5)
            top_tokens = []
            
            for idx, weight in zip(top_k.indices, top_k.values):
                token_text = tokenizer.decode([idx.item()])
                top_tokens.append(f"{token_text}({weight:.3f})")
            
            print(f"Position {pos}: '{original_text}' -> {', '.join(top_tokens)}")
        
    return {
        'vocab_weights': vocab_weights,
        'input_ids': input_ids,
        'tokenizer': tokenizer
    }

# Simple usage
def main():
    result = simple_knowledge_graph_extraction(
        "outputs/sym_4gpu_final/checkpoint_epoch_4.pt",
        "The cat sat on the mat"
    )
    print("\n✅ Done!")

if __name__ == "__main__":
    main()