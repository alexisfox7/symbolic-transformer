import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

def get_token_through_first_layer(model, token_id, position_idx, seq_length, device='cuda'):
    """Pass a single token through the first transformer layer."""
    pad_token_id = 50256
    dummy_input = torch.full((1, seq_length), pad_token_id, dtype=torch.long, device=device)
    dummy_input[0, position_idx] = token_id
    
    with torch.no_grad():
        inputs_embeds = model.transformer.wte(dummy_input)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        first_block = model.transformer.h[0]
        attention_mask = torch.zeros((1, seq_length), device=device)
        attention_mask[0, position_idx] = 1.0
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        outputs = first_block(hidden_states, attention_mask=extended_attention_mask)
        return outputs[0][0, position_idx, :]

def test_magnitude_correction(model, text, device='cpu'):
    """Test magnitude correction across layers."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    
    # Test just a few layers and positions for speed
    test_layers = [0, 3, 6, 9]
    position = 1  # Test middle position
    k = 3
    
    for layer_idx in test_layers:
        print(f"\nLAYER {layer_idx}, Position {position} ('{tokens[position]}')")
        print("-" * 60)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            residual = outputs.hidden_states[layer_idx + 1][0, position, :]
            
            # Get top k tokens via LogitLens
            residual_normed = model.transformer.ln_f(residual.unsqueeze(0))
            logits = model.lm_head(residual_normed)[0]
            topk_values, topk_indices = torch.topk(logits, k=k)
            
            # Get raw and transformed embeddings
            raw_embeddings = model.transformer.wte(topk_indices)
            
            transformed_embeddings = []
            for token_id in topk_indices:
                transformed = get_token_through_first_layer(
                    model, token_id, position, input_ids.shape[1], device
                )
                transformed_embeddings.append(transformed)
            transformed_embeddings = torch.stack(transformed_embeddings)
            
            # === ORIGINAL METHOD (magnitude-biased) ===
            A_raw = raw_embeddings.T
            A_trans = transformed_embeddings.T
            b = residual.unsqueeze(1)
            
            coeffs_raw = torch.linalg.lstsq(A_raw, b).solution.squeeze()
            coeffs_trans = torch.linalg.lstsq(A_trans, b).solution.squeeze()
            
            recon_raw = (A_raw @ coeffs_raw.unsqueeze(1)).squeeze()
            recon_trans = (A_trans @ coeffs_trans.unsqueeze(1)).squeeze()
            
            error_raw = torch.norm(residual - recon_raw).item()
            error_trans = torch.norm(residual - recon_trans).item()
            
            # === MAGNITUDE-CORRECTED METHOD ===
            residual_norm = torch.norm(residual)
            residual_normalized = residual / residual_norm
            
            raw_norms = torch.norm(raw_embeddings, dim=1, keepdim=True)
            trans_norms = torch.norm(transformed_embeddings, dim=1, keepdim=True)
            
            raw_embeddings_normalized = raw_embeddings / (raw_norms + 1e-8)
            transformed_embeddings_normalized = transformed_embeddings / (trans_norms + 1e-8)
            
            A_raw_norm = raw_embeddings_normalized.T
            A_trans_norm = transformed_embeddings_normalized.T
            b_norm = residual_normalized.unsqueeze(1)
            
            coeffs_raw_norm = torch.linalg.lstsq(A_raw_norm, b_norm).solution.squeeze()
            coeffs_trans_norm = torch.linalg.lstsq(A_trans_norm, b_norm).solution.squeeze()
            
            recon_raw_norm = (A_raw_norm @ coeffs_raw_norm.unsqueeze(1)).squeeze() * residual_norm
            recon_trans_norm = (A_trans_norm @ coeffs_trans_norm.unsqueeze(1)).squeeze() * residual_norm
            
            error_raw_norm = torch.norm(residual - recon_raw_norm).item()
            error_trans_norm = torch.norm(residual - recon_trans_norm).item()
            
            # Calculate percentages
            residual_magnitude = residual_norm.item()
            error_raw_pct = 100 * error_raw / residual_magnitude
            error_trans_pct = 100 * error_trans / residual_magnitude
            error_raw_norm_pct = 100 * error_raw_norm / residual_magnitude
            error_trans_norm_pct = 100 * error_trans_norm / residual_magnitude
            
            improvement_original = error_raw_pct - error_trans_pct
            improvement_corrected = error_raw_norm_pct - error_trans_norm_pct
            
            print(f"Target residual norm: {residual_magnitude:.1f}")
            print(f"Top tokens: {[tokenizer.decode([idx.item()]) for idx in topk_indices]}")
            
            print(f"\n=== ORIGINAL (magnitude-biased) ===")
            print(f"Raw embedding norms: {[f'{n.item():.1f}' for n in raw_norms.squeeze()]}")
            print(f"Transformed norms: {[f'{n.item():.1f}' for n in trans_norms.squeeze()]}")
            print(f"Raw error: {error_raw:.1f} ({error_raw_pct:.1f}%)")
            print(f"Trans error: {error_trans:.1f} ({error_trans_pct:.1f}%)")
            print(f"Improvement: {improvement_original:.1f} pp")
            
            print(f"\n=== MAGNITUDE-CORRECTED ===")
            print(f"Raw coefficients: {coeffs_raw_norm.cpu().numpy()}")
            print(f"Trans coefficients: {coeffs_trans_norm.cpu().numpy()}")
            print(f"Raw error: {error_raw_norm:.1f} ({error_raw_norm_pct:.1f}%)")
            print(f"Trans error: {error_trans_norm:.1f} ({error_trans_norm_pct:.1f}%)")
            print(f"Improvement: {improvement_corrected:.1f} pp")
            
            if abs(improvement_corrected) < 1:
                print("→ Magnitude correction removes the improvement!")
            elif improvement_corrected > 0:
                print("→ Real directional improvement confirmed")
            else:
                print("→ First-layer transformation actually hurts direction")

def main():
    device = 'cpu'  # Use CPU for speed
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The girl was named"
    test_magnitude_correction(model, text, device)

if __name__ == "__main__":
    main()