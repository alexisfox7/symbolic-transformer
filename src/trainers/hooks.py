#src/trainers/hooks.py
"""
Hook system for trainers, inspired by TransformerLens.
"""

from typing import Dict, Any, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingHook:
    """
    Simple training hook base class.
    
    Similar to TransformerLens hooks but for training events instead of model activations.
    Each hook method receives the current trainer state and can read/modify it.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    def __repr__(self):
        return f"TrainingHook('{self.name}', enabled={self.enabled})"
    
    # core training events 
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        """Called once at start of training."""
        pass
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        """Called once at end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, state: Dict[str, Any]) -> None:
        """Called at start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        """Called at end of each epoch."""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        """Called after each training batch."""
        pass


class HookManager:
    """
    Manages and calls training hooks.
    
    Design principles:
    1. Simple registration/removal
    2. Safe execution (one hook failure doesn't break training)
    3. Clear logging of what's happening
    4. Minimal overhead when no hooks are registered
    """
    
    def __init__(self):
        self.hooks: List[TrainingHook] = []
        self.logger = logging.getLogger(__name__)
    
    def add_hook(self, hook: TrainingHook) -> None:
        """Add a hook. Replaces any existing hook with the same name."""
        # Remove existing hook with same name
        self.hooks = [h for h in self.hooks if h.name != hook.name]
        self.hooks.append(hook)
        self.logger.info(f"Added hook: {hook.name}")
    
    def remove_hook(self, name: str) -> bool:
        """Remove hook by name. Returns True if found and removed."""
        original_len = len(self.hooks)
        self.hooks = [h for h in self.hooks if h.name != name]
        removed = len(self.hooks) < original_len
        if removed:
            self.logger.info(f"Removed hook: {name}")
        return removed
    
    def get_hook(self, name: str) -> Optional[TrainingHook]:
        """Get hook by name."""
        for hook in self.hooks:
            if hook.name == name:
                return hook
        return None
    
    def enable_hook(self, name: str) -> bool:
        """Enable a hook by name."""
        hook = self.get_hook(name)
        if hook:
            hook.enabled = True
            return True
        return False
    
    def disable_hook(self, name: str) -> bool:
        """Disable a hook by name."""
        hook = self.get_hook(name)
        if hook:
            hook.enabled = False
            return True
        return False
    
    def list_hooks(self) -> List[str]:
        """List all hook names."""
        return [h.name for h in self.hooks]
    
    def _call_hook_method(self, method_name: str, *args, **kwargs) -> None:
        """
        Safely call a method on all enabled hooks.
        
        Each hook runs in isolation - if one fails, others still run.
        """
        if not self.hooks:
            return  # Fast path when no hooks
        
        for hook in self.hooks:
            if not hook.enabled:
                continue
            
            try:
                method = getattr(hook, method_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook {hook.name}.{method_name} failed: {e}", exc_info=True)
                # Continue with other hooks
    
    # Convenience methods for trainer to call
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_train_begin', state)
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_train_end', state)
    
    def on_epoch_begin(self, epoch: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_epoch_begin', epoch, state)
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_epoch_end', epoch, state)
    
    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_batch_end', batch_idx, loss, state)


class ConsoleLogHook(TrainingHook):
    """Console logging hook."""
    
    def __init__(self, log_every_n_batches: int = 10):
        super().__init__("console_log")
        self.log_every_n_batches = log_every_n_batches
        self.logger = logging.getLogger(__name__)
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        epochs = state.get('num_epochs', '?')
        model_params = state.get('model_params', '?')
        self.logger.info(f"Training started: {epochs} epochs, {model_params:,} parameters")
    
    def on_epoch_begin(self, epoch: int, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        total = state.get('num_epochs', '?')
        self.logger.info(f"Epoch {epoch}/{total}")
    
    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        if batch_idx % self.log_every_n_batches == 0:
            if not state.get('is_main_process', True):
                return
            epoch = state.get('current_epoch', '?')
            self.logger.info(f"  Batch {batch_idx}, Loss: {loss:.4f}")
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        avg_loss = state.get('avg_loss', 'N/A')
        duration = state.get('epoch_duration', 'N/A')
        if isinstance(avg_loss, float):
            self.logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}, time={duration:.1f}s")
        else:
            self.logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss}, time={duration}s")


class JSONLogHook(TrainingHook):
    """JSON logging hook."""
    
    def __init__(self, output_dir: str, log_every_n_batches: int = 100):
        super().__init__("json_log")
        self.log_every_n_batches = log_every_n_batches
        
        import os
        import json
        from datetime import datetime
        
        # Setup log file
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, "logs", f"training_{timestamp}.jsonl")
        
        # Write header
        self._write({
            "event": "log_start",
            "timestamp": datetime.now().isoformat(),
            "log_file": self.log_file
        })
    
    def _write(self, data: Dict[str, Any]) -> None:
        """Write JSON line to log file."""
        import json
        from datetime import datetime
        
        data["timestamp"] = datetime.now().isoformat()
        
        try:
            with open(self.log_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception:
            pass  # Silent fail
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        config = {k: v for k, v in state.items() 
                 if not callable(v) and k not in ['model', 'optimizer', 'dataloader', 'accelerator']}
        self._write({"event": "train_begin", "config": config})
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        self._write({
            "event": "epoch_end",
            "epoch": epoch,
            "avg_loss": state.get('avg_loss'),
            "duration": state.get('epoch_duration')
        })
    
    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        if batch_idx % self.log_every_n_batches == 0:
            if not state.get('is_main_process', True):
                return
            self._write({
                "event": "batch",
                "epoch": state.get('current_epoch', 0),
                "batch": batch_idx,
                "loss": loss
            })


class CheckpointHook(TrainingHook):
    """Checkpointing hook."""
    
    def __init__(self, output_dir: str, save_every_n_epochs: int = 1):
        super().__init__("checkpoint")
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if epoch % self.save_every_n_epochs == 0:
            if not state.get('is_main_process', True):
                return
                
            import os
            import torch
            
            model = state.get('model')
            optimizer = state.get('optimizer')
            
            if model and optimizer and self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': state.get('avg_loss'),
                }
                
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                
                logger.info(f"Saved checkpoint: {checkpoint_path}")


# Factory functions for easy hook creation
def create_console_log_hook(log_every_n_batches: int = 10) -> ConsoleLogHook:
    """Create a console logging hook."""
    return ConsoleLogHook(log_every_n_batches)


def create_json_log_hook(output_dir: str, log_every_n_batches: int = 100) -> JSONLogHook:
    """Create a JSON logging hook."""
    return JSONLogHook(output_dir, log_every_n_batches)


def create_checkpoint_hook(output_dir: str, save_every_n_epochs: int = 1) -> CheckpointHook:
    """Create a checkpointing hook."""
    return CheckpointHook(output_dir, save_every_n_epochs)


class ValidationHook(TrainingHook):
    """Validation hook."""
    
    def __init__(self, val_dataloader, device, validate_every=1, model_type=""):
        super().__init__("validation")
        self.val_dataloader = val_dataloader
        self.device = device
        self.validate_every = validate_every
        self.model_type = model_type
        self.enabled = True
        
    def on_epoch_end(self, epoch, state):
        if not self.enabled or not self.val_dataloader:
            return
            
        if epoch % self.validate_every == 0:
            model = state.get('model')
            if model:
                from utils.training_utils import run_validation
                val_metrics = run_validation(model, self.val_dataloader, self.device)
                prefix = f"{self.model_type} " if self.model_type else ""
                logger.info(
                    f"{prefix}Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"Perplexity: {val_metrics['perplexity']:.2f}"
                )