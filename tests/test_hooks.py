#!/usr/bin/env python3
"""
Test script for JSONLogHook and ConsoleLogHook
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Initialize accelerate state to avoid logger errors
from accelerate import PartialState
import logging
logging.basicConfig(level=logging.INFO)
_ = PartialState()

from hooks.training import JSONLogHook, ConsoleLogHook
from hooks.base import HookManager

def test_hooks():
    """Test both JSONLogHook and ConsoleLogHook"""
    
    # Create temp directory for JSON logs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing hooks with temp dir: {temp_dir}")
        
        # Create hooks
        console_hook = ConsoleLogHook(log_every_n_batches=2)
        json_hook = JSONLogHook(output_dir=temp_dir, log_every_n_batches=2)
        
        # Create hook manager
        hook_manager = HookManager()
        hook_manager.add_hook(console_hook)
        hook_manager.add_hook(json_hook)
        
        print("Created hooks and hook manager")
        
        # Mock training state
        mock_state = {
            'num_epochs': 3,
            'model_params': 1000000,
            'is_main_process': True,
            'current_epoch': 1,
            'current_batch_idx': 0,
            'latest_loss': 2.5,
            'avg_epoch_loss': 2.3,
            'epoch_duration': 45.2,
            'status': 'Completed',
            'training_time': 120.5,
            'total_batches': 100,
            'final_loss': 2.1,
            'val_loss': 2.0,
            'val_perplexity': 7.39
        }
        
        print("\n=== Testing Training Begin ===")
        hook_manager.call_hooks('on_train_begin', mock_state)
        
        print("\n=== Testing Epoch Begin ===")
        hook_manager.call_hooks('on_epoch_begin', mock_state)
        
        print("\n=== Testing Batch Begin ===")
        hook_manager.call_hooks('on_batch_begin', mock_state)
        
        print("\n=== Testing Batch End (batch 0) ===")
        hook_manager.call_hooks('on_batch_end', mock_state)
        
        # Test batch 1 (shouldn't log due to log_every_n_batches=2)
        mock_state['current_batch_idx'] = 1
        mock_state['latest_loss'] = 2.4
        print("\n=== Testing Batch End (batch 1 - should not log) ===")
        hook_manager.call_hooks('on_batch_end', mock_state)
        
        # Test batch 2 (should log)
        mock_state['current_batch_idx'] = 2
        mock_state['latest_loss'] = 2.3
        print("\n=== Testing Batch End (batch 2 - should log) ===")
        hook_manager.call_hooks('on_batch_end', mock_state)
        
        print("\n=== Testing Epoch End ===")
        hook_manager.call_hooks('on_epoch_end', mock_state)
        
        print("\n=== Testing Training End ===")
        hook_manager.call_hooks('on_train_end', mock_state)
        
        # Check JSON log file
        json_files = [f for f in os.listdir(temp_dir) if f.startswith('training_') and f.endswith('.jsonl')]
        
        if json_files:
            json_file_path = os.path.join(temp_dir, json_files[0])
            print(f"\n=== JSON Log File Contents ({json_files[0]}) ===")
            
            with open(json_file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        print(f"Line {i}: {data['event']} - {data.get('timestamp', 'N/A')}")
                        if data['event'] == 'batch':
                            print(f"  Batch {data['batch']}: loss={data['loss']}, perplexity={data['perplexity']:.2f}")
                        elif data['event'] == 'epoch_end':
                            print(f"  Epoch {data['epoch']}: loss={data['loss']}, duration={data['duration']}s")
                            if 'val_loss' in data:
                                print(f"  Validation: loss={data['val_loss']}, perplexity={data['val_perplexity']:.2f}")
                    except json.JSONDecodeError as e:
                        print(f"Line {i}: JSON decode error - {e}")
                        print(f"  Raw line: {line.strip()}")
        else:
            print("\n❌ No JSON log file found!")
            
        print(f"\n✅ Hook test completed!")

if __name__ == "__main__":
    test_hooks()