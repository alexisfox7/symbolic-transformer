#!/usr/bin/env python3
"""
Simple test for attention analysis script
"""

import json
import torch
from attention import AttentionTester, load_model_from_checkpoint, load_tokenizer


def create_dummy_checkpoint():
    """Create a minimal dummy checkpoint for testing"""
    from src.config import TransformerConfig
    from src.model import get_model
    
    # Create a tiny config for testing
    config = TransformerConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    
    model = get_model('vanilla', config)
    
    # Save dummy checkpoint
    checkpoint = {
        'config': config,
        'model_state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, 'test_checkpoint.pt')
    return 'test_checkpoint.pt', config


def test_basic_functionality():
    """Test basic functionality with a small example"""
    print("Creating test checkpoint...")
    checkpoint_path, config = create_dummy_checkpoint()
    
    print("Testing with small dataset...")
    # Create test data similar to pair_data.json format
    test_data = [
        {
            "id": "test_001_distractor_A_1",
            "target_story": "test_001_A", 
            "target_sentence": "The boy was happy.",
            "context_sentences": [
                "The boy found a toy",  # Relevant
                "He picked it up",      # Relevant  
                "The girl saw a cat",   # Irrelevant
                "She petted it"         # Irrelevant
            ],
            "distractor_source": "test_001_B"
        }
    ]
    
    # Save test data
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Load model
        device = 'cpu'
        model, config = load_model_from_checkpoint(checkpoint_path, device, 'vanilla')
        tokenizer = load_tokenizer('gpt2')
        
        print("Testing attention analysis...")
        tester = AttentionTester('test_data.json', device)
        
        # Test text concatenation
        example = test_data[0]
        full_text = tester._concatenate_sentences(example)
        print(f"Concatenated text: {full_text}")
        
        # Test attention extraction (simplified)
        print("Testing attention extraction...")
        # This might fail due to model complexity, but let's see basic structure
        
        results = tester.test_model(model, tokenizer, 'test_vanilla', max_examples=1)
        print(f"Basic test results: {results['stats']}")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import os
        for f in ['test_checkpoint.pt', 'test_data.json']:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    test_basic_functionality()