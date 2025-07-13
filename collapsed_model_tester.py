#!/usr/bin/env python3
"""
Test how badly a rank-collapsed model performs.
Shows the real-world impact of rank collapse on model behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Run with: python -m collapsed_model_tester [args] or set PYTHONPATH before running

def load_model_and_tokenizer(checkpoint_path, model_type="vanilla"):
    """Load model and tokenizer from checkpoint."""
    from src.model import get_model
    from src.config import TransformerConfig
    from src.mytokenizers import create_tokenizer
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        if hasattr(config_data, '__dict__'):
            config = config_data
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    # Create model and tokenizer
    model = get_model(model_type, config)
    tokenizer = create_tokenizer('gpt2')
    config.update_from_tokenizer(tokenizer)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded: {model.get_num_params()/1e6:.2f}M parameters")
    
    return model, tokenizer, config

def test_output_similarity(model, tokenizer):
    """Test if different inputs produce similar outputs (sign of collapse)."""
    print("\n" + "="*60)
    print("🧪 TESTING OUTPUT SIMILARITY")
    print("="*60)
    
    model.eval()
    
    # Test 1: Completely different token sequences
    test_cases = [
        ("The cat sat on", [1, 2, 3, 4]),
        ("Python programming is", [100, 200, 300, 400]),
        ("Beautiful sunny day", [500, 600, 700, 800]),
        ("Machine learning models", [1000, 1100, 1200, 1300]),
        ("Random words here", [2000, 2100, 2200, 2300]),
    ]
    
    outputs = []
    with torch.no_grad():
        for description, token_ids in test_cases:
            # Ensure tokens are within vocab range
            max_vocab = min(tokenizer.vocab_size, 10000)  # Cap at reasonable size
            safe_tokens = [min(t, max_vocab-1) for t in token_ids]
            
            input_tensor = torch.tensor([safe_tokens])
            output = model(input_tensor)
            outputs.append((description, output['logits']))
    
    # Compute pairwise similarities
    print("Pairwise output similarities:")
    similarities = []
    
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            desc1, logits1 = outputs[i]
            desc2, logits2 = outputs[j]
            
            # Compare the last token's logits (next token prediction)
            sim = F.cosine_similarity(
                logits1[0, -1, :].unsqueeze(0),  # Last position
                logits2[0, -1, :].unsqueeze(0), 
                dim=1
            ).item()
            
            similarities.append(sim)
            print(f"  '{desc1}' vs '{desc2}': {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    max_similarity = np.max(similarities)
    
    print(f"\nSummary:")
    print(f"  Average similarity: {avg_similarity:.4f}")
    print(f"  Maximum similarity: {max_similarity:.4f}")
    
    # Assessment
    if avg_similarity > 0.9:
        print(f"🚨 SEVERE: Model outputs are nearly identical!")
        health = "BROKEN"
    elif avg_similarity > 0.7:
        print(f"⚠️ MODERATE: Model outputs are too similar")
        health = "POOR" 
    elif avg_similarity > 0.5:
        print(f"⚠️ MILD: Some similarity but might work")
        health = "CONCERNING"
    else:
        print(f"✅ GOOD: Model produces diverse outputs")
        health = "HEALTHY"
    
    return health, avg_similarity, max_similarity

def test_generation_quality(model, tokenizer):
    """Test actual text generation quality."""
    print("\n" + "="*60)
    print("📝 TESTING GENERATION QUALITY")
    print("="*60)
    
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "The weather today is",
        "My favorite food is"
    ]
    
    results = []
    
    for prompt in prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        
        try:
            # Encode prompt
            inputs = tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    inputs, 
                    max_new_tokens=10, 
                    temperature=0.8,
                    tokenizer=tokenizer
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            continuation = generated_text[len(prompt):].strip()
            
            print(f"   Generated: '{continuation}'")
            results.append((prompt, continuation))
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            results.append((prompt, "[FAILED]"))
    
    # Analyze generation quality
    print(f"\n📊 Generation Analysis:")
    
    # Check for repetition
    repetitive_count = 0
    empty_count = 0
    error_count = 0
    
    for prompt, continuation in results:
        if continuation == "[FAILED]":
            error_count += 1
        elif len(continuation.strip()) == 0:
            empty_count += 1
        elif len(set(continuation.split())) <= 2:  # Very few unique words
            repetitive_count += 1
            print(f"   🔄 Repetitive: '{prompt}' → '{continuation}'")
    
    total_tests = len(results)
    print(f"   Successful generations: {total_tests - error_count}/{total_tests}")
    print(f"   Empty outputs: {empty_count}/{total_tests}")
    print(f"   Repetitive outputs: {repetitive_count}/{total_tests}")
    
    # Health assessment
    if error_count > total_tests // 2:
        gen_health = "BROKEN"
        print(f"🚨 Generation is BROKEN (many failures)")
    elif repetitive_count > total_tests // 2:
        gen_health = "POOR"
        print(f"⚠️ Generation is POOR (very repetitive)")
    elif empty_count > total_tests // 3:
        gen_health = "CONCERNING"
        print(f"⚠️ Generation is CONCERNING (many empty outputs)")
    else:
        gen_health = "OK"
        print(f"✅ Generation appears to work")
    
    return gen_health, results

def test_embedding_diversity(model):
    """Test how diverse the learned embeddings are."""
    print("\n" + "="*60)
    print("🎭 TESTING EMBEDDING DIVERSITY")
    print("="*60)
    
    # Get embeddings
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embeddings = model.transformer.wte.weight.data
    else:
        print("❌ Cannot find embeddings")
        return "UNKNOWN"
    
    vocab_size, n_embd = embeddings.shape
    print(f"Embedding matrix: {vocab_size} × {n_embd}")
    
    # Sample some embeddings to test
    sample_size = min(100, vocab_size)
    sample_indices = torch.randperm(vocab_size)[:sample_size]
    sample_embeddings = embeddings[sample_indices]
    
    # Test 1: Pairwise distances
    distances = torch.cdist(sample_embeddings, sample_embeddings)
    # Remove diagonal
    mask = ~torch.eye(sample_size, dtype=torch.bool)
    off_diag_distances = distances[mask]
    
    mean_distance = off_diag_distances.mean().item()
    min_distance = off_diag_distances.min().item()
    
    print(f"Token embedding distances (sample of {sample_size}):")
    print(f"  Mean distance: {mean_distance:.4f}")
    print(f"  Min distance: {min_distance:.4f}")
    
    # Test 2: Effective dimensionality
    # SVD on sample
    U, S, V = torch.svd(sample_embeddings)
    effective_rank = (S.sum() / S[0]).item()
    
    print(f"Effective dimensionality: {effective_rank:.1f} / {n_embd}")
    
    # Assessment
    if mean_distance < 0.1:
        emb_health = "COLLAPSED"
        print(f"🚨 Embeddings are COLLAPSED (very close together)")
    elif mean_distance < 0.5:
        emb_health = "POOR"
        print(f"⚠️ Embeddings are too similar") 
    elif effective_rank < n_embd * 0.1:
        emb_health = "LOW_RANK"
        print(f"⚠️ Embeddings are low-rank")
    else:
        emb_health = "HEALTHY"
        print(f"✅ Embeddings appear diverse")
    
    return emb_health

def comprehensive_collapse_test(checkpoint_path, model_type="vanilla"):
    """Run comprehensive test of collapsed model."""
    print("🔬 COMPREHENSIVE RANK COLLAPSE IMPACT TEST")
    print("="*80)
    
    try:
        # Load model
        model, tokenizer, config = load_model_and_tokenizer(checkpoint_path, model_type)
        
        # Run tests
        print(f"Model type: {model_type}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
        
        # Test 1: Output similarity
        sim_health, avg_sim, max_sim = test_output_similarity(model, tokenizer)
        
        # Test 2: Generation quality  
        gen_health, gen_results = test_generation_quality(model, tokenizer)
        
        # Test 3: Embedding diversity
        emb_health = test_embedding_diversity(model)
        
        # Overall assessment
        print("\n" + "="*80)
        print("📋 OVERALL ASSESSMENT")
        print("="*80)
        
        print(f"Output Similarity Health: {sim_health}")
        print(f"Generation Health: {gen_health}")
        print(f"Embedding Health: {emb_health}")
        
        # Determine if model is usable
        broken_indicators = [sim_health, gen_health, emb_health]
        severe_issues = broken_indicators.count("BROKEN") + broken_indicators.count("COLLAPSED")
        moderate_issues = broken_indicators.count("POOR") + broken_indicators.count("LOW_RANK")
        
        print(f"\nSevere issues: {severe_issues}/3")
        print(f"Moderate issues: {moderate_issues}/3")
        
        if severe_issues >= 2:
            overall = "🚨 MODEL IS SEVERELY BROKEN"
            recommendation = "Must retrain with rank collapse fixes"
        elif severe_issues >= 1 or moderate_issues >= 2:
            overall = "⚠️ MODEL HAS SERIOUS ISSUES"
            recommendation = "Strongly recommend retraining"
        elif moderate_issues >= 1:
            overall = "⚠️ MODEL HAS SOME ISSUES"
            recommendation = "Consider retraining for better performance"
        else:
            overall = "✅ MODEL APPEARS FUNCTIONAL"
            recommendation = "Model should work adequately"
        
        print(f"\n{overall}")
        print(f"Recommendation: {recommendation}")
        
        return {
            'overall_health': overall,
            'similarity_health': sim_health,
            'generation_health': gen_health,
            'embedding_health': emb_health,
            'avg_output_similarity': avg_sim,
            'recommendation': recommendation
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test rank-collapsed model performance')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--model-type', type=str, default='vanilla', 
                       choices=['vanilla', 'symbolic', 'tft'])
    
    args = parser.parse_args()
    
    comprehensive_collapse_test(args.checkpoint, args.model_type)

if __name__ == "__main__":
    main()