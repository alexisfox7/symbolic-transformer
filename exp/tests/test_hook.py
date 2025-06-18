#!/usr/bin/env python
# test_hook_system.py
"""
Integration test for the hook system across all trainers.
Tests that hooks work consistently with SimpleTrainer and AccelerateTrainer.
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from trainers import get_trainer
from trainers.hooks import TrainingHook, create_console_log_hook, create_json_log_hook, create_checkpoint_hook
from model.architectures.vanilla import VanillaTransformer
from config.config import TransformerConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TestHook(TrainingHook):
    """Custom test hook to verify hook system functionality."""
    
    def __init__(self):
        super().__init__("test_hook")
        self.events_called = []
        self.batch_losses = []
        self.epoch_losses = []
        
    def on_train_begin(self, state):
        self.events_called.append(('train_begin', state.get('num_epochs', 'unknown')))
        logger.info(f"🎯 TestHook: Training started with {state.get('num_epochs', '?')} epochs")
        
    def on_train_end(self, state):
        self.events_called.append(('train_end', state.get('status', 'unknown')))
        logger.info(f"🎯 TestHook: Training ended with status: {state.get('status', 'unknown')}")
        
    def on_epoch_begin(self, epoch, state):
        self.events_called.append(('epoch_begin', epoch))
        logger.info(f"🎯 TestHook: Epoch {epoch} started")
        
    def on_epoch_end(self, epoch, state):
        loss = state.get('loss', state.get('avg_loss', 'unknown'))
        self.events_called.append(('epoch_end', epoch, loss))
        self.epoch_losses.append(loss)
        logger.info(f"🎯 TestHook: Epoch {epoch} ended with loss: {loss}")
        
    def on_batch_end(self, batch_idx, loss, state):
        self.events_called.append(('batch_end', batch_idx, loss))
        self.batch_losses.append(loss)
        if batch_idx <= 2:  # Only log first few batches to avoid spam
            logger.info(f"🎯 TestHook: Batch {batch_idx} ended with loss: {loss:.4f}")
    
    def get_summary(self):
        """Get a summary of hook activity."""
        summary = {
            'total_events': len(self.events_called),
            'train_begin_count': len([e for e in self.events_called if e[0] == 'train_begin']),
            'train_end_count': len([e for e in self.events_called if e[0] == 'train_end']),
            'epoch_begin_count': len([e for e in self.events_called if e[0] == 'epoch_begin']),
            'epoch_end_count': len([e for e in self.events_called if e[0] == 'epoch_end']),
            'batch_end_count': len([e for e in self.events_called if e[0] == 'batch_end']),
            'total_batches_seen': len(self.batch_losses),
            'total_epochs_seen': len(self.epoch_losses),
            'avg_batch_loss': sum(self.batch_losses) / len(self.batch_losses) if self.batch_losses else 0,
            'final_epoch_loss': self.epoch_losses[-1] if self.epoch_losses else None
        }
        return summary

def create_test_model():
    """Create a VanillaTransformer model for testing."""
    config = TransformerConfig(
        vocab_size=50,  # Small vocab for testing
        n_layer=2,      # Few layers for fast testing
        n_head=2,       # Few heads for testing
        n_embd=32,      # Small embedding dimension
        block_size=16,  # Small sequence length
        dropout=0.1,
        bias=False
    )
    return VanillaTransformer(config)

def create_test_data(num_samples=32, seq_len=16, vocab_size=50, batch_size=8):
    """Create test dataset for language modeling."""
    # Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    # For language modeling, targets are the same as inputs
    targets = input_ids.clone()
    
    # Create dataset that returns dicts
    dataset = []
    for i in range(num_samples):
        dataset.append({
            'input_ids': input_ids[i],
            'targets': targets[i]
        })
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_trainer_with_hooks(trainer_type, test_name):
    """Test a specific trainer type with the hook system."""
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 TESTING {test_name}")
    logger.info(f"{'='*60}")
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp(prefix=f"hook_test_{trainer_type}_")
    
    try:
        # Setup test components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_test_model()
        dataloader = create_test_data(num_samples=24, seq_len=16, vocab_size=50, batch_size=4)  # 6 batches per epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        logger.info(f"📊 Test setup: {len(dataloader)} batches per epoch, 2 epochs")
        logger.info(f"🖥️  Device: {device}")
        
        # Create trainer
        if trainer_type == "simple":
            trainer = get_trainer(
                trainer_type="simple",
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_epochs=2,
                output_dir=temp_dir,
                log_interval=2
            )
        elif trainer_type == "accelerate":
            trainer = get_trainer(
                trainer_type="accelerate",
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_epochs=2,
                output_dir=temp_dir,
                log_interval=2
            )
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")
        
        # Add hooks
        test_hook = TestHook()
        trainer.add_hook(test_hook)
        
        # Add built-in hooks
        trainer.add_console_logging(log_every_n_batches=3)
        trainer.add_json_logging(log_every_n_batches=2)
        trainer.add_checkpointing(save_every_n_epochs=1)
        
        # Verify hooks were added
        hook_names = trainer.hooks.list_hooks()
        logger.info(f"🔗 Added hooks: {hook_names}")
        assert "test_hook" in hook_names
        assert "console_log" in hook_names
        assert "json_log" in hook_names
        assert "checkpoint" in hook_names
        
        # Test hook management
        trainer.hooks.disable_hook("console_log")
        console_hook = trainer.get_hook("console_log")
        assert console_hook.enabled == False
        
        trainer.hooks.enable_hook("console_log")
        assert console_hook.enabled == True
        
        # Run training
        logger.info("🚀 Starting training...")
        result = trainer.train()
        
        # Verify training completed
        assert result is not None
        assert isinstance(result, dict)
        logger.info(f"✅ Training result: {result}")
        
        # Verify hook functionality
        summary = test_hook.get_summary()
        logger.info(f"📈 Hook Summary: {summary}")
        
        # Expected counts for 2 epochs with 6 batches each
        expected_batch_count = 12  # 6 batches × 2 epochs
        expected_epoch_count = 2
        
        # Assertions
        assert summary['train_begin_count'] == 1, f"Expected 1 train_begin, got {summary['train_begin_count']}"
        assert summary['train_end_count'] == 1, f"Expected 1 train_end, got {summary['train_end_count']}"
        assert summary['epoch_begin_count'] == expected_epoch_count, f"Expected {expected_epoch_count} epoch_begin, got {summary['epoch_begin_count']}"
        assert summary['epoch_end_count'] == expected_epoch_count, f"Expected {expected_epoch_count} epoch_end, got {summary['epoch_end_count']}"
        assert summary['batch_end_count'] == expected_batch_count, f"Expected {expected_batch_count} batch_end, got {summary['batch_end_count']}"
        assert summary['total_epochs_seen'] == expected_epoch_count
        
        # Verify checkpoints were created
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('checkpoint_epoch_')]
        logger.info(f"💾 Checkpoints created: {checkpoint_files}")
        assert len(checkpoint_files) >= 1, "Expected at least 1 checkpoint file"
        
        # Verify JSON logs were created
        log_dir = os.path.join(temp_dir, 'logs')
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.jsonl')]
            logger.info(f"📝 JSON log files: {log_files}")
            # Note: JSON logs might not be created in distributed training on non-main processes
        
        logger.info(f"✅ {test_name} PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {test_name} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def test_hook_manager_directly():
    """Test HookManager functionality directly."""
    logger.info(f"\n{'='*60}")
    logger.info("🧪 TESTING HOOK MANAGER DIRECTLY")
    logger.info(f"{'='*60}")
    
    from trainers.hooks import HookManager
    
    manager = HookManager()
    test_hook = TestHook()
    
    # Test adding hooks
    manager.add_hook(test_hook)
    assert "test_hook" in manager.list_hooks()
    
    # Test getting hooks
    retrieved_hook = manager.get_hook("test_hook")
    assert retrieved_hook is test_hook
    
    # Test enabling/disabling
    manager.disable_hook("test_hook")
    assert not test_hook.enabled
    
    manager.enable_hook("test_hook")
    assert test_hook.enabled
    
    # Test hook calls
    test_state = {'test': 'data'}
    manager.on_train_begin(test_state)
    manager.on_epoch_begin(1, test_state)
    manager.on_batch_end(0, 0.5, test_state)
    manager.on_epoch_end(1, test_state)
    manager.on_train_end(test_state)
    
    # Verify events were recorded
    summary = test_hook.get_summary()
    assert summary['train_begin_count'] == 1
    assert summary['epoch_begin_count'] == 1
    assert summary['batch_end_count'] == 1
    assert summary['epoch_end_count'] == 1
    assert summary['train_end_count'] == 1
    
    # Test removing hooks
    manager.remove_hook("test_hook")
    assert "test_hook" not in manager.list_hooks()
    
    logger.info("✅ HOOK MANAGER TESTS PASSED!")
    return True

def main():
    """Run all hook system tests."""
    logger.info("🎯 STARTING HOOK SYSTEM INTEGRATION TESTS")
    
    results = {}
    
    # Test HookManager directly
    try:
        results['hook_manager'] = test_hook_manager_directly()
    except Exception as e:
        logger.error(f"Hook manager test failed: {e}")
        results['hook_manager'] = False
    
    # Test SimpleTrainer with hooks
    try:
        results['simple_trainer'] = test_trainer_with_hooks("simple", "SIMPLE TRAINER WITH HOOKS")
    except Exception as e:
        logger.error(f"SimpleTrainer test failed: {e}")
        results['simple_trainer'] = False
    
    # Test AccelerateTrainer with hooks (if available)
    try:
        results['accelerate_trainer'] = test_trainer_with_hooks("accelerate", "ACCELERATE TRAINER WITH HOOKS")
    except Exception as e:
        logger.error(f"AccelerateTrainer test failed: {e}")
        results['accelerate_trainer'] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL HOOK SYSTEM TESTS PASSED!")
        logger.info("Hook system is fully integrated and working correctly!")
        return True
    else:
        logger.error(f"💥 {total_tests - passed_tests} tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)