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
    from model import get_model
    
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

        # Now let's try vocab grounding more carefully
        print(f"\n=== VOCAB GROUNDING - CAREFUL APPROACH ===")
        
        # Apply vocab grounding to the hidden states
        vocab_grounded = model.vocab_grounding(hidden_states)
        print(f"Vocab grounded output shape: {vocab_grounded.shape}")
        
        # Let's manually look at just one channel to debug
        print(f"\n=== DEBUGGING ONE CHANNEL ===")
        
        # Get the hidden states reshaped for channels
        B, T, C = hidden_states.shape
        n_head = model.vocab_grounding.n_head
        head_dim = model.vocab_grounding.head_dim
        
        print(f"Hidden: {hidden_states.shape}, n_head: {n_head}, head_dim: {head_dim}")
        
        # Reshape to channels
        hidden_channels = hidden_states.view(B, T, n_head, head_dim)
        
        # Just look at channel 0 for token 1 (" cat")
        h = 0  # First channel
        pos = 1  # " cat" token
        
        hidden_h = hidden_channels[:, :, h, :]  # (B, T, head_dim)
        print(f"Channel {h} hidden shape: {hidden_h.shape}")
        
        # Apply channel FFN
        ffn_h = model.vocab_grounding.channel_ffns[h](hidden_h)
        print(f"After channel FFN: {ffn_h.shape}")
        print(f"FFN output sample: {ffn_h[0, pos, :5]}")  # First 5 values
        
        # Get vocab attention logits
        vocab_logits_h = model.vocab_grounding.channel_vocab_attentions[h](ffn_h)
        print(f"Vocab logits shape: {vocab_logits_h.shape}")
        print(f"Vocab logits for position {pos} ('{tokens[pos]}') sample: {vocab_logits_h[0, pos, :10]}")
        
        # Check temperature
        temp_h = torch.clamp(model.vocab_grounding.channel_temperatures[h], min=0.1)
        print(f"Channel {h} temperature: {temp_h}")
        
        # Get vocab weights
        vocab_weights_h = F.softmax(vocab_logits_h / temp_h, dim=-1)
        print(f"Vocab weights shape: {vocab_weights_h.shape}")
        
        # Check if weights are reasonable
        pos_weights = vocab_weights_h[0, pos]  # Weights for " cat" token
        top_weights = torch.topk(pos_weights, k=10)
        
        print(f"\nTop 10 vocab weights for '{tokens[pos]}' in channel {h}:")
        for idx, weight in zip(top_weights.indices, top_weights.values):
            token_text = tokenizer.decode([idx.item()])
            print(f"  '{token_text}': {weight:.6f}")
        
        # Check if weights sum to 1
        weight_sum = pos_weights.sum()
        print(f"Weight sum: {weight_sum:.6f} (should be ~1.0)")
        
        # Now let's check reconstruction quality
        print(f"\n=== RECONSTRUCTION CHECK ===")
        print("Comparing ln(ffn(x)) with vocab projected version")
        
        # Get the hidden states just before vocab grounding
        original_hidden = hidden_states.clone()  # (1, 6, 384)
        
        # Apply vocab grounding
        vocab_projected = model.vocab_grounding(hidden_states)  # (1, 6, 384)
        
        print(f"Original hidden shape: {original_hidden.shape}")
        print(f"Vocab projected shape: {vocab_projected.shape}")
        
        # Compare reconstruction quality token by token
        for pos in range(len(tokens)):
            token = tokens[pos]
            
            orig_vec = original_hidden[0, pos]  # (384,)
            proj_vec = vocab_projected[0, pos]  # (384,)
            
            # Compute reconstruction metrics
            mse_loss = F.mse_loss(proj_vec, orig_vec)
            cosine_sim = F.cosine_similarity(orig_vec.unsqueeze(0), proj_vec.unsqueeze(0))
            l2_norm_orig = torch.norm(orig_vec)
            l2_norm_proj = torch.norm(proj_vec)
            
            print(f"Position {pos} '{token}':")
            print(f"  MSE Loss: {mse_loss:.6f}")
            print(f"  Cosine Similarity: {cosine_sim:.6f}")
            print(f"  L2 Norm - Original: {l2_norm_orig:.6f}, Projected: {l2_norm_proj:.6f}")
            print(f"  Norm Ratio: {l2_norm_proj/l2_norm_orig:.6f}")
        
        # Overall reconstruction quality
        total_mse = F.mse_loss(vocab_projected, original_hidden)
        total_cosine = F.cosine_similarity(
            original_hidden.view(-1, 384), 
            vocab_projected.view(-1, 384)
        ).mean()
        
        print(f"\n=== OVERALL RECONSTRUCTION ===")
        print(f"Total MSE: {total_mse:.6f}")
        print(f"Average Cosine Similarity: {total_cosine:.6f}")
        
        # Check if reconstruction is meaningful
        if total_mse > 1.0:
            print("⚠️  HIGH MSE: Vocab projection changes representations significantly")
            print("   This suggests vocab projection might not be preserving semantics")
        
        if total_cosine < 0.8:
            print("⚠️  LOW COSINE SIMILARITY: Projected vectors point in different directions")
            print("   This suggests vocab projection is not a good reconstruction")
        
        if total_cosine > 0.95 and total_mse < 0.1:
            print("✅ GOOD RECONSTRUCTION: Vocab projection preserves original semantics")
            print("   The vocabulary decompositions should be meaningful")
        
        # Check if vocab projection is just identity
        identity_check = torch.allclose(original_hidden, vocab_projected, atol=1e-3)
        if identity_check:
            print("⚠️  IDENTITY MAPPING: Vocab projection barely changes the input")
            print("   This suggests the vocab constraint isn't being enforced")
        
        # Let's also check what happens if we bypass vocab grounding
        print(f"\n=== BYPASS CHECK ===")
        
        # Get final layer output WITHOUT vocab grounding
        lm_logits_original = model.lm_head(original_hidden)
        lm_logits_projected = model.lm_head(vocab_projected)
        
        # Compare predictions
        orig_preds = torch.argmax(lm_logits_original, dim=-1)
        proj_preds = torch.argmax(lm_logits_projected, dim=-1)
        
        print("Predictions - Original vs Vocab Projected:")
        for pos in range(len(tokens)):
            orig_pred = tokenizer.decode([orig_preds[0, pos].item()])
            proj_pred = tokenizer.decode([proj_preds[0, pos].item()])
            same = "✓" if orig_pred == proj_pred else "✗"
            print(f"  Position {pos}: '{orig_pred}' vs '{proj_pred}' {same}")

def main():
    simple_vocab_analysis(
        "outputs/sym_4gpu_final/checkpoint_epoch_4.pt",
        "The cat sat on the mat"
    )

if __name__ == "__main__":
    main()