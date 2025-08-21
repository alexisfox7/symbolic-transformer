import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_subtract_sanity():
    """Test the subtraction logic with known vectors."""
    
    # Create simple test vectors
    residual = torch.tensor([1.0, 2.0, 3.0])  # Original residual
    token_embed = torch.tensor([0.5, 1.0, 1.5])  # Token embedding (same direction, scaled)
    
    print("=== Test 1: Same direction vectors ===")
    print(f"Residual: {residual}")
    print(f"Token embed: {token_embed}")
    
    # Calculate optimal scale (should be positive)
    scale = torch.dot(residual, token_embed) / torch.dot(token_embed, token_embed)
    print(f"Optimal scale: {scale:.3f}")
    
    # Subtract
    residual_after = residual - scale * token_embed
    print(f"After subtraction: {residual_after}")
    
    # Check what was actually subtracted
    subtracted = residual - residual_after
    print(f"What was subtracted: {subtracted}")
    print(f"Expected (scale * token_embed): {scale * token_embed}")
    
    # Calculate cosine similarity (WRONG way - like in the code)
    cosine_wrong = F.cosine_similarity(subtracted.unsqueeze(0), token_embed.unsqueeze(0)).item()
    print(f"Cosine similarity (wrong way): {cosine_wrong:.3f}")
    
    # Calculate cosine similarity (RIGHT way - between original vectors)
    cosine_right = F.cosine_similarity(residual.unsqueeze(0), token_embed.unsqueeze(0)).item()
    print(f"Cosine similarity (right way): {cosine_right:.3f}")
    
    print("\n=== Test 2: Opposite direction vectors ===")
    residual2 = torch.tensor([1.0, 2.0, 3.0])
    token_embed2 = torch.tensor([-0.5, -1.0, -1.5])  # Opposite direction
    
    print(f"Residual: {residual2}")
    print(f"Token embed: {token_embed2}")
    
    scale2 = torch.dot(residual2, token_embed2) / torch.dot(token_embed2, token_embed2)
    print(f"Optimal scale: {scale2:.3f}")  # Should be negative
    
    residual_after2 = residual2 - scale2 * token_embed2
    subtracted2 = residual2 - residual_after2
    
    cosine_wrong2 = F.cosine_similarity(subtracted2.unsqueeze(0), token_embed2.unsqueeze(0)).item()
    cosine_right2 = F.cosine_similarity(residual2.unsqueeze(0), token_embed2.unsqueeze(0)).item()
    
    print(f"Cosine similarity (wrong way): {cosine_wrong2:.3f}")  # Should be -1
    print(f"Cosine similarity (right way): {cosine_right2:.3f}")  # Should be -1
    
    print("\n=== Test 3: GPT2 actual case ===")
    # Test with actual GPT2 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Get actual residual
    text = "The cat sat on the mat"
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        residual = outputs.hidden_states[5][0, 2, :]  # layer 5, position 2 ("sat")
        
        # Get " cat" token embedding
        cat_token_id = tokenizer.encode(" cat", add_special_tokens=False)[0]
        cat_embed = model.transformer.wte(torch.tensor([cat_token_id], device=device))[0]
        
    print(f"Residual norm: {torch.norm(residual).item():.3f}")
    print(f"Cat embed norm: {torch.norm(cat_embed).item():.3f}")
    
    # Calculate cosine similarity between original vectors
    true_cosine = F.cosine_similarity(residual.unsqueeze(0), cat_embed.unsqueeze(0)).item()
    print(f"True cosine similarity (residual vs cat_embed): {true_cosine:.3f}")
    
    # Calculate optimal scale
    scale_actual = torch.dot(residual, cat_embed) / torch.dot(cat_embed, cat_embed)
    print(f"Optimal scale: {scale_actual.item():.3f}")
    
    # This explains the behavior: if true cosine similarity is negative,
    # the optimal scale will be negative, making the "wrong" cosine calculation = -1

if __name__ == "__main__":
    test_subtract_sanity()