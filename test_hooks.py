#!/usr/bin/env python3
"""
Test script for inference hooks integration.
Tests that hooks are properly called during model generation.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import (
    InferenceHook, 
    InferenceHookManager,
    create_attention_extraction_hook,
    AttentionExtractionHook,
    SymbolicStreamHook,
    ActivationHook
)
from src.mytokenizers import create_tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHook(InferenceHook):
    """Simple test hook that counts method calls."""
    
    def __init__(self):
        super().__init__("test_hook")
        self.call_counts = {
            'generation_begin': 0,
            'generation_end': 0,
            'forward_begin': 0,
            'forward_end': 0,
            'attention_computed': 0,
            'ffn_computed': 0
        }
    
    def on_generation_begin(self, prompt_tokens, state):
        self.call_counts['generation_begin'] += 1
        logger.info(f"Generation starting with {len(prompt_tokens)} tokens")
    
    def on_generation_end(self, generated_tokens, state):
        self.call_counts['generation_end'] += 1
        logger.info(f"Generation ended with {len(generated_tokens)} tokens")
    
    def on_forward_begin(self, input_ids, position, state):
        self.call_counts['forward_begin'] += 1
    
    def on_forward_end(self, logits, position, state):
        self.call_counts['forward_end'] += 1
    
    def on_attention_computed(self, layer_idx, head_idx, attention_weights, 
                            query, key, value, tokens, position, state):
        self.call_counts['attention_computed'] += 1
    
    def on_ffn_computed(self, layer_idx, ffn_input, ffn_output, tokens, position, state):
        self.call_counts['ffn_computed'] += 1


def test_basic_hooks():
    """Test that hooks are called during generation."""
    print("\n=== Testing Basic Hook Integration ===")
    
    # Create small model for testing
    config = TransformerConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('vanilla', config).to(device)
    model.eval()
    
    # Get tokenizer
    tokenizer = create_tokenizer('character')
    
    # Create test hook
    test_hook = TestHook()
    
    # Run generation with hook
    prompt = "Hello"
    max_new_tokens = 5
    
    ids, text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        show_progress=False,
        hooks=[test_hook]
    )
    
    # Verify hooks were called
    print(f"\nGenerated text: {text}")
    print(f"\nHook call counts:")
    for method, count in test_hook.call_counts.items():
        print(f"  {method}: {count}")
    
    # Assertions
    assert test_hook.call_counts['generation_begin'] == 1, "generation_begin should be called once"
    assert test_hook.call_counts['generation_end'] == 1, "generation_end should be called once"
    assert test_hook.call_counts['forward_begin'] == max_new_tokens, f"forward_begin should be called {max_new_tokens} times"
    assert test_hook.call_counts['forward_end'] == max_new_tokens, f"forward_end should be called {max_new_tokens} times"
    
    expected_attention_calls = max_new_tokens * config.n_layer * config.n_head
    assert test_hook.call_counts['attention_computed'] == expected_attention_calls, \
        f"attention_computed should be called {expected_attention_calls} times"
    
    expected_ffn_calls = max_new_tokens * config.n_layer
    assert test_hook.call_counts['ffn_computed'] == expected_ffn_calls, \
        f"ffn_computed should be called {expected_ffn_calls} times"
    
    print("\n✓ Basic hook test passed!")


def test_attention_extraction_hook():
    """Test the attention extraction hook."""
    print("\n=== Testing Attention Extraction Hook ===")
    
    # Create small model
    config = TransformerConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('vanilla', config).to(device)
    model.eval()
    
    tokenizer = create_tokenizer('character')
    
    # Create attention extraction hook
    attention_hook = create_attention_extraction_hook(threshold=0.1, store_values=False)
    
    # Run generation
    prompt = "Test"
    max_new_tokens = 3
    
    ids, text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        show_progress=False,
        hooks=[attention_hook]
    )
    
    print(f"\nGenerated text: {text}")
    print(f"Number of attention records: {len(attention_hook.attention_data)}")
    
    # Check that we have attention data
    assert len(attention_hook.attention_data) > 0, "Should have attention data"
    
    # Examine first attention record
    first_record = attention_hook.attention_data[0]
    print(f"\nFirst attention record:")
    print(f"  Layer: {first_record['layer']}")
    print(f"  Head: {first_record['head']}")
    print(f"  Position: {first_record['position']}")
    print(f"  Number of edges: {len(first_record['edges'])}")
    
    # Test edge extraction
    if first_record['edges']:
        edge = first_record['edges'][0]
        print(f"\nExample edge:")
        print(f"  {edge['source_token']} -> {edge['target_token']}")
        print(f"  Weight: {edge['weight']:.4f}")
    
    # Test token attention summary
    if len(text) > 0:
        first_char = text[0]
        summary = attention_hook.get_token_attention_summary(first_char)
        print(f"\nAttention summary for '{first_char}':")
        print(f"  Total received: {summary['total_received']:.4f}")
        print(f"  Total given: {summary['total_given']:.4f}")
    
    print("\n✓ Attention extraction hook test passed!")


def test_symbolic_model_hooks():
    """Test hooks with symbolic transformer model."""
    print("\n=== Testing Symbolic Model Hooks ===")
    
    # Create symbolic model
    config = TransformerConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('symbolic', config).to(device)
    model.eval()
    
    tokenizer = create_tokenizer('character')
    
    # Create hooks
    test_hook = TestHook()
    symbolic_hook = SymbolicStreamHook()
    
    # Run generation
    prompt = "Hi"
    max_new_tokens = 3
    
    ids, text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        show_progress=False,
        hooks=[test_hook, symbolic_hook]
    )
    
    print(f"\nGenerated text: {text}")
    print(f"Stream data records: {len(symbolic_hook.stream_data)}")
    
    # Check symbolic stream tracking
    if symbolic_hook.stream_data:
        first_stream = symbolic_hook.stream_data[0]
        print(f"\nFirst stream record:")
        print(f"  Stream type: {first_stream.get('stream_type', 'unknown')}")
    
    print("\n✓ Symbolic model hook test passed!")


def test_multiple_hooks():
    """Test running multiple hooks simultaneously."""
    print("\n=== Testing Multiple Hooks ===")
    
    config = TransformerConfig(
        vocab_size=1000,
        n_embd=128,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('vanilla', config).to(device)
    model.eval()
    
    tokenizer = create_tokenizer('character')
    
    # Create multiple hooks
    test_hook = TestHook()
    attention_hook = AttentionExtractionHook(threshold=0.2)
    activation_hook = ActivationHook(layers_to_track=[0, 1])
    
    # Run generation
    prompt = "A"
    max_new_tokens = 2
    
    ids, text = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        show_progress=False,
        hooks=[test_hook, attention_hook, activation_hook]
    )
    
    print(f"\nGenerated text: {text}")
    print(f"Test hook forward calls: {test_hook.call_counts['forward_begin']}")
    print(f"Attention records: {len(attention_hook.attention_data)}")
    print(f"Activation records: {len(activation_hook.activations)}")
    
    # Verify all hooks were active
    assert test_hook.call_counts['forward_begin'] > 0
    assert len(attention_hook.attention_data) > 0
    assert len(activation_hook.activations) > 0
    
    print("\n✓ Multiple hooks test passed!")


def test_hook_manager_directly():
    """Test the hook manager directly."""
    print("\n=== Testing Hook Manager ===")
    
    manager = InferenceHookManager()
    
    # Add hooks
    hook1 = TestHook()
    hook1.name = "hook1"
    hook2 = TestHook()
    hook2.name = "hook2"
    
    manager.add_hook(hook1)
    manager.add_hook(hook2)
    
    # Test listing hooks
    hook_names = manager.list_hooks()
    print(f"Registered hooks: {hook_names}")
    assert "hook1" in hook_names
    assert "hook2" in hook_names
    
    # Test calling methods
    manager.on_generation_begin(["test"], {})
    assert hook1.call_counts['generation_begin'] == 1
    assert hook2.call_counts['generation_begin'] == 1
    
    # Test removing hook
    manager.remove_hook("hook1")
    assert "hook1" not in manager.list_hooks()
    
    # Test disabling hook
    hook2.enabled = False
    manager.on_generation_begin(["test"], {})
    assert hook2.call_counts['generation_begin'] == 1  # Should not increase
    
    print("\n✓ Hook manager test passed!")


if __name__ == "__main__":
    try:
        test_hook_manager_directly()
        test_basic_hooks()
        test_attention_extraction_hook()
        test_symbolic_model_hooks()
        test_multiple_hooks()
        
        print("\n\n🎉 All tests passed! Hook integration is working correctly.")
        
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)