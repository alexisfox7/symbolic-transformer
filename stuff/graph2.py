import torch
import torch.nn.functional as F
import os
import sys

def simple_vocab_analysis(checkpoint_path: str, text: str):
    """
    Simple vocab analysis that actually works with your checkpoint structure.
    """
    # Add parent directory for imports
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from config import get_preset_config
    from mytokenizers import create_tokenizer
    from modelold import get_model
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint - we know the structure now
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    
    # Create config for medium model (384 dims)
    config = get_preset_config('medium')
    
    # Create tokenizer and update config
    tokenizer = create_tokenizer('gpt2')
    config.update_from_tokenizer(tokenizer)
    
    print(f"Config: n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}")
    
    # Create and load model
    model = get_model("Symbolic", config=config)
    model.load_state_dict(model_state_dict)  # No prefix stripping needed
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = [tokenizer.decode([t]) for t in input_ids.squeeze()]
    
    print(f"\nInput: '{text}'")
    print(f"Tokens: {tokens}")
    
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids)
        
        print(f"\n=== MODEL PREDICTIONS ===")
        print(f"Output logits shape: {outputs['logits'].shape}")
        
        # Show what the model actually predicts (this should be reasonable)
        logits = outputs['logits'][0]  # (seq_len, vocab_size)
        
        for pos in range(len(tokens)):
            current_token = tokens[pos]
            
            # Get top 3 predictions for NEXT token
            top_logits = torch.topk(logits[pos], k=3)
            predictions = []
            
            for idx, score in zip(top_logits.indices, top_logits.values):
                pred_text = tokenizer.decode([idx.item()])
                predictions.append(f"'{pred_text}'")
            
            print(f"After '{current_token}' -> predicts: {', '.join(predictions)}")
        
        # Now let's check the vocab grounding layer specifically
        print(f"\n=== VOCAB GROUNDING ANALYSIS ===")
        
        # Check vocab grounding layer structure
        vg = model.vocab_grounding
        print(f"Vocab grounding type: {type(vg)}")
        print(f"Has channel_vocab_attentions: {hasattr(vg, 'channel_vocab_attentions')}")
        
        if hasattr(vg, 'channel_vocab_attentions'):
            print(f"Number of channels: {len(vg.channel_vocab_attentions)}")
            print(f"Channel vocab attention shapes: {[list(layer.weight.shape) for layer in vg.channel_vocab_attentions]}")
        
        # Try a much simpler approach - just look at final layer norm output
        print(f"\n=== SIMPLE APPROACH ===")
        
        # Get hidden states just before the language model head
        hidden_states = model.transformer.wte(input_ids)  # Token embeddings
        
        # Pass through transformer layers
        for layer in model.transformer.h:
            hidden_states = layer(hidden_states)
        
        # Final layer norm
        hidden_states = model.transformer.ln_f(hidden_states)
        
        print(f"Final hidden states shape: {hidden_states.shape}")
        
        # Get the language model head predictions (this should work)
        lm_logits = model.lm_head(hidden_states)  # (1, seq_len, vocab_size)
        lm_probs = F.softmax(lm_logits, dim=-1).squeeze(0)  # (seq_len, vocab_size)
        
        print(f"Language model probabilities shape: {lm_probs.shape}")
        
        # Show top tokens from language model head (this should be reasonable)
        print(f"\nTop tokens from language model head:")
        for pos in range(len(tokens)):
            current_token = tokens[pos]
            
            # Get top 5 most likely NEXT tokens from LM head
            top_probs = torch.topk(lm_probs[pos], k=5)
            top_tokens = []
            
            for idx, prob in zip(top_probs.indices, top_probs.values):
                token_text = tokenizer.decode([idx.item()])
                top_tokens.append(f"'{token_text}'({prob:.3f})")
            
            print(f"Position {pos} '{current_token}' -> {', '.join(top_tokens)}")

def main():
    simple_vocab_analysis(
        "outputs/sym_4gpu_final/checkpoint_epoch_4.pt",
        "The cat sat on the mat"
    )

if __name__ == "__main__":
    main()