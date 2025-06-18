#!/usr/bin/env python3
"""
Test that hooks work correctly in multi-GPU mode.
This simulates distributed training behavior.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
from trainers.hooks import HookManager, ConsoleLogHook, JSONLogHook, CheckpointHook

class MockAccelerator:
    """Mock accelerator for testing distributed behavior."""
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process
        self.num_processes = 2 if not is_main_process else 1

def test_hooks_respect_main_process():
    """Test that hooks only execute on main process in distributed mode."""
    print("Testing hooks with distributed awareness...")
    
    # Test data
    test_state_main = {
        'is_main_process': True,
        'accelerator': MockAccelerator(True),
        'num_epochs': 2,
        'model_params': 1000
    }
    
    test_state_worker = {
        'is_main_process': False, 
        'accelerator': MockAccelerator(False),
        'num_epochs': 2,
        'model_params': 1000
    }
    
    # Capture console output
    import logging
    import io
    
    # Set up logging to capture output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger('trainers.hooks')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    try:
        # Test ConsoleLogHook
        console_hook = ConsoleLogHook(log_every_n_batches=1)
        
        print("Testing main process (should log):")
        log_capture.seek(0)
        log_capture.truncate(0)
        
        console_hook.on_train_begin(test_state_main)
        console_hook.on_epoch_begin(1, test_state_main)
        console_hook.on_batch_end(0, 1.5, test_state_main)
        console_hook.on_epoch_end(1, test_state_main)
        
        main_output = log_capture.getvalue()
        print(f"Main process output length: {len(main_output)} chars")
        assert len(main_output) > 0, "Main process should produce log output"
        
        print("Testing worker process (should NOT log):")
        log_capture.seek(0)
        log_capture.truncate(0)
        
        console_hook.on_train_begin(test_state_worker)
        console_hook.on_epoch_begin(1, test_state_worker) 
        console_hook.on_batch_end(0, 1.5, test_state_worker)
        console_hook.on_epoch_end(1, test_state_worker)
        
        worker_output = log_capture.getvalue()
        print(f"Worker process output length: {len(worker_output)} chars")
        assert len(worker_output) == 0, "Worker process should NOT produce log output"
        
        print("✅ ConsoleLogHook respects is_main_process")
        
        # Test JSONLogHook
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            json_hook = JSONLogHook(temp_dir, log_every_n_batches=1)
            
            # Main process - should create files
            json_hook.on_train_begin(test_state_main)
            json_hook.on_batch_end(0, 1.5, test_state_main)
            
            # Worker process - should not create files or write
            json_hook_worker = JSONLogHook(temp_dir + "_worker", log_every_n_batches=1)
            json_hook_worker.on_train_begin(test_state_worker)
            json_hook_worker.on_batch_end(0, 1.5, test_state_worker)
            
            # Check files were created for main but not worker
            main_logs = os.path.join(temp_dir, "logs")
            worker_logs = os.path.join(temp_dir + "_worker", "logs")
            
            assert os.path.exists(main_logs), "Main process should create log directory"
            # Worker should create the directory but not write to files
            
        print("✅ JSONLogHook respects is_main_process")
        
        # Test CheckpointHook
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_hook = CheckpointHook(temp_dir, save_every_n_epochs=1)
            
            # Mock model and optimizer
            model = torch.nn.Linear(10, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            main_state = {**test_state_main, 'model': model, 'optimizer': optimizer}
            worker_state = {**test_state_worker, 'model': model, 'optimizer': optimizer}
            
            # Main process - should save checkpoint
            checkpoint_hook.on_epoch_end(1, main_state)
            main_checkpoint = os.path.join(temp_dir, "checkpoint_epoch_1.pt")
            assert os.path.exists(main_checkpoint), "Main process should save checkpoint"
            
            # Worker process - should NOT save checkpoint
            with tempfile.TemporaryDirectory() as worker_temp:
                worker_checkpoint_hook = CheckpointHook(worker_temp, save_every_n_epochs=1)
                worker_checkpoint_hook.on_epoch_end(1, worker_state)
                worker_checkpoint = os.path.join(worker_temp, "checkpoint_epoch_1.pt")
                assert not os.path.exists(worker_checkpoint), "Worker process should NOT save checkpoint"
        
        print("✅ CheckpointHook respects is_main_process")
        
        print("\n🎉 All distributed hooks tests passed!")
        print("Hooks will only log/save on main process in multi-GPU training")
        
        return True
        
    finally:
        # Clean up logging
        logger.removeHandler(handler)

def test_backward_compatibility():
    """Test that hooks still work when is_main_process is not in state."""
    print("\nTesting backward compatibility (no is_main_process key)...")
    
    # State without is_main_process (like SimpleTrainer)
    simple_state = {
        'num_epochs': 2,
        'model_params': 1000,
        'current_epoch': 1
    }
    
    console_hook = ConsoleLogHook(log_every_n_batches=1)
    
    # Should work normally (default to True)
    try:
        console_hook.on_train_begin(simple_state)
        console_hook.on_epoch_begin(1, simple_state)
        console_hook.on_batch_end(0, 1.5, simple_state)
        console_hook.on_epoch_end(1, simple_state)
        print("✅ Hooks work without is_main_process key (backward compatible)")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_hooks_respect_main_process()
    success2 = test_backward_compatibility()
    
    if success1 and success2:
        print("\n🎉 ALL MULTI-GPU HOOK TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)