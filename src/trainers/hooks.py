#src/trainers/hooks.py
"""
Enhanced hook system with perplexity calculation and validation JSON logging.
"""

from typing import Dict, Any, List, Callable, Optional
import logging
import torch
import math
from random import randint
import torch.nn as nn

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
    
    def on_batch_begin(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        """Called after each training batch."""
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
    
    def remove_hook(self, name: str) -> bool:
        """Remove hook by name. Returns True if found and removed."""
        original_len = len(self.hooks)
        self.hooks = [h for h in self.hooks if h.name != name]
        removed = len(self.hooks) < original_len
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
    
    def on_batch_begin(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_batch_begin', batch_idx, loss, state)

    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_batch_end', batch_idx, loss, state)


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from loss, handling edge cases."""
    if loss is None or math.isnan(loss) or math.isinf(loss):
        return float('nan')
    
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


class ConsoleLogHook(TrainingHook):
    """Console logging hook with perplexity."""
    
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
            perplexity = calculate_perplexity(loss)
            self.logger.info(f"  Batch {batch_idx}, Loss: {loss:.4f}, Perplexity: {perplexity:.2f}")
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        avg_loss = state.get('avg_loss', state.get('loss', 'N/A'))
        duration = state.get('epoch_duration', 'N/A')
        
        if isinstance(avg_loss, (int, float)) and not math.isnan(avg_loss):
            perplexity = calculate_perplexity(avg_loss)
            self.logger.info(f"Epoch {epoch} complete: loss={avg_loss:.4f}, perplexity={perplexity:.2f}, time={duration:.1f}s")
        else:
            self.logger.info(f"Epoch {epoch} complete: loss={avg_loss}, time={duration}s")


class JSONLogHook(TrainingHook):
    """JSON logging hook with perplexity and validation metrics."""
    
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
        
        # Training metrics
        avg_loss = state.get('avg_loss', state.get('loss'))
        epoch_data = {
            "event": "epoch_end",
            "epoch": epoch,
            "loss": avg_loss,
            "duration": state.get('epoch_duration')
        }
        
        # Add perplexity if loss is available
        if avg_loss is not None:
            epoch_data["perplexity"] = calculate_perplexity(avg_loss)
        
        # Add validation metrics if available
        val_loss = state.get('val_loss')
        val_perplexity = state.get('val_perplexity')
        if val_loss is not None:
            epoch_data["val_loss"] = val_loss
            epoch_data["val_perplexity"] = val_perplexity or calculate_perplexity(val_loss)
        
        self._write(epoch_data)
    
    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        if batch_idx % self.log_every_n_batches == 0:
            if not state.get('is_main_process', True):
                return
            
            batch_data = {
                "event": "batch",
                "epoch": state.get('current_epoch', 0),
                "batch": batch_idx,
                "loss": loss,
                "perplexity": calculate_perplexity(loss)
            }
            self._write(batch_data)


class CheckpointHook(TrainingHook):
    """Enhanced checkpointing hook that saves config and comprehensive training state."""
    
    def __init__(self, output_dir: str, save_every_n_epochs: int = 1, save_best: bool = True):
        super().__init__("checkpoint")
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best = save_best
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, state: Dict[str, Any]) -> None:
        if epoch % self.save_every_n_epochs == 0:
            if not state.get('is_main_process', True):
                return
                
            import os
            import torch
            
            model = state.get('model')
            optimizer = state.get('optimizer')
            config = getattr(model, 'config', None)  # Get config from model if available
            
            if model and optimizer and self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': state.get('avg_loss', state.get('loss')),
                }
                
                # Add config if available (from model or trainer state)
                if config is not None:
                    checkpoint['config'] = config
                elif 'config' in state:
                    checkpoint['config'] = state['config']
                
                # Add validation metrics if available
                if 'val_loss' in state:
                    checkpoint['val_loss'] = state['val_loss']
                    checkpoint['val_perplexity'] = state.get('val_perplexity')
                
                # Add training metrics from state
                training_metrics = {}
                metric_keys = ['epoch_losses', 'total_batches', 'total_samples', 'training_time']
                for key in metric_keys:
                    if key in state:
                        training_metrics[key] = state[key]
                
                if training_metrics:
                    checkpoint['training_metrics'] = training_metrics
                
                # Add any extra data from state
                extra_keys = ['symbolic_features', 'model_params']
                for key in extra_keys:
                    if key in state:
                        checkpoint[key] = state[key]
                
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
                
                logger.info(f"Enhanced checkpoint saved: {checkpoint_path}")
                if config is not None:
                    logger.info(f"  Included config: {type(config).__name__}")
                if training_metrics:
                    logger.info(f"  Included metrics: {list(training_metrics.keys())}")
                
                # Save best model checkpoint if enabled
                if self.save_best:
                    current_loss = state.get('val_loss', state.get('avg_loss', state.get('loss', float('inf'))))
                    if isinstance(current_loss, (int, float)) and current_loss < self.best_loss:
                        self.best_loss = current_loss
                        best_path = os.path.join(self.output_dir, "best_model.pt")
                        checkpoint['best_loss'] = self.best_loss
                        torch.save(checkpoint, best_path)
                        logger.info(f"Saved best model (loss={self.best_loss:.4f}): {best_path}")

class ValidationHook(TrainingHook):
    """Validation hook that adds metrics to state for other hooks to use."""
    
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
                
                # Add validation metrics to state for other hooks to use
                state['val_loss'] = val_metrics['loss']
                state['val_perplexity'] = val_metrics['perplexity']
                state['val_samples'] = val_metrics['samples']
                
                # Console logging
                prefix = f"{self.model_type} " if self.model_type else ""
                logger.info(
                    f"{prefix}Validation - Loss: {val_metrics['loss']:.4f}, "
                    f"Perplexity: {val_metrics['perplexity']:.2f}"
                )


class EarlyExitHook(TrainingHook):
    """
    Adds auxillary loss from randomly selecting layer output each batch to decode and test.
    """

    def __init__(self):
        super().__init__("early_exit")
        # Initialize attributes to avoid AttributeError
        self.aux_loss = None
        self.exit_layer_output = None
        self.random_layer_idx = None
        self.lm_head = None
        self.layer_norm = None
        self.targets = None

    def on_batch_begin(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        model = state.get('model')
        self.model = model  # Store model reference
            
        self.random_layer_idx = randint(0, model.config.n_layer-1)
        self.exit_layer_output = None
        self.aux_loss = None  # Reset aux loss for this batch
        self.lm_head = model.lm_head
        self.layer_norm = model.transformer.ln_f
        batch_data = state.get('current_batch', {})
        if 'targets' in batch_data:
            self.targets = batch_data['targets'].clone()
        elif 'input_ids' in batch_data:
            self.targets = batch_data['input_ids'].clone()
        else:
            raise ValueError("No targets or input_ids found in batch")
    
    def analyze_layer(self, hidden_state, layer_idx: int, position, tokens):
        "choose randomized layer, decode output"
        if layer_idx == self.random_layer_idx:
            self.exit_layer_output = hidden_state.clone() # (B, T, n_embd)
            # Compute aux loss immediately after capture
            self._compute_aux_loss()
        
        # Also set a fallback zero loss if this is the last layer and nothing was captured
        # This handles the case where random_layer_idx is beyond actual layers
        if hasattr(self, 'model') and self.model and layer_idx == self.model.config.n_layer - 1:
            if self.aux_loss is None:
                device = hidden_state.device
                self.aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
    def _compute_aux_loss(self):
        """Compute auxiliary loss from captured layer output."""
        if self.exit_layer_output is None:
            return
        
        # Apply layer norm and compute logits
        # For TFT models, we might need different processing
        normalized_output = self.layer_norm(self.exit_layer_output)
        logits = self.lm_head(normalized_output) # (B, T, vocab_size)

        loss_func = nn.CrossEntropyLoss() # expects (N, C) , (N)

        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_target = self.targets[:, 1:].contiguous().view(-1)
        
        aux_loss = loss_func(shift_logits, shift_target)
        self.aux_loss = aux_loss

    def on_batch_end(self, batch_idx: int, loss: float, state: Dict[str, Any]) -> None:
        # Aux loss should already be computed in analyze_layer
        # But if no layer was captured (random layer outside range), set zero loss
        if self.aux_loss is None:
            device = state.get('device', torch.device('cpu'))
            self.aux_loss = torch.tensor(0.0, device=device, requires_grad=True)

    def get_aux_loss(self):
        "Return auxiliary loss + what type it is"
        if self.aux_loss is None:
            raise ValueError("Auxiliary loss is empty")
        
        info = {'loss_type': 'early_exit'}

        return self.aux_loss, info




        


