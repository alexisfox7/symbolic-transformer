from typing import Any, Dict

from accelerate.logging import get_logger # type: ignore
from .base import TrainingHook
import os
import json  
import math
from datetime import datetime
logger = get_logger(__name__)

class ConsoleLogHook(TrainingHook):
    def __init__(self, log_every_n_batches: int):
        super().__init__("console_log")
        self.log_every_n_batches = log_every_n_batches

    def on_train_begin(self, state: Dict[str, Any]) -> None:
        epochs = state.get('num_epochs', '?')
        model_params = state.get('model_params', '?')
        logger.info(f"Training started: {epochs} epochs, {model_params:,} parameters")
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        logger.info("Training ended!")
    
    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        epoch_idx = state.get("current_epoch")
        total = state.get("num_epochs")
        logger.info(f"Epoch {epoch_idx}/{total}")

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        epoch_idx = state.get("current_epoch")
        avg_loss = state.get("avg_epoch_loss", 0.0)
        epoch_duration = state.get("epoch_duration", 0.0)
        logger.info(f"Epoch {epoch_idx} completed: avg_loss={avg_loss:.4f}, duration={epoch_duration:.2f}s")
    
    def on_batch_begin(self, state: Dict[str, Any]) -> None:
        batch_idx = state.get("current_batch_idx")
        current_epoch = state.get("current_epoch")
        total_epochs = state.get("num_epochs")

        if batch_idx % self.log_every_n_batches == 0:
            logger.info(f"Epoch {current_epoch}/{total_epochs}, Batch {batch_idx}")

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        batch_idx = state.get("current_batch_idx")
        loss = state.get("latest_loss")
        if batch_idx % self.log_every_n_batches == 0:
            current_epoch = state.get("current_epoch")
            total_epochs = state.get("num_epochs")
            logger.info(f"Epoch {current_epoch}/{total_epochs}, Batch {batch_idx} completed: loss={loss:.4f}")

class JSONLogHook(TrainingHook):
    def __init__(self, output_dir: str, log_every_n_batches: int = 100):
        super().__init__("json_log")
        self.log_every_n_batches = log_every_n_batches
        
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"training_{timestamp}.jsonl"
        self.log_file = os.path.join(output_dir, filename)

        initial_data = {"event": "train_start", "log_file": self.log_file}
        self._write(initial_data)

    def _calc_perplexity(self, loss):
        return math.exp(loss)
    
    def _write(self, data):
        data["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')
    
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        config = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                 for k, v in state.items() 
                 if not callable(v) and k not in ['model', 'optimizer', 'dataloader', 'accelerator', 'current_batch']}
        self._write({"event": "train_begin", "config": config})
    
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        
        # training metrics
        avg_loss = state.get('avg_epoch_loss', state.get('loss'))
        epoch_data = {
            "event": "epoch_end",
            "epoch": state.get('current_epoch'),
            "loss": avg_loss,
            "duration": state.get('epoch_duration')
        }

        epoch_data["perplexity"] = self._calc_perplexity(avg_loss)
        
        # add validation metrics if available
        val_loss = state.get('val_loss')
        val_perplexity = state.get('val_perplexity')
        if val_loss is not None:
            epoch_data["val_loss"] = val_loss
            epoch_data["val_perplexity"] = val_perplexity or self._calc_perplexity(val_loss)
        
        self._write(epoch_data)
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        if not state.get('is_main_process', True):
            return
        
        final_data = {
            "event": "train_end",
            "status": state.get('status', 'Unknown'),
            "training_time": state.get('training_time', 0.0),
            "total_batches": state.get('total_batches', 0),
            "final_loss": state.get('final_loss')
        }
        self._write(final_data)
    
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        batch_idx = state.get("current_batch_idx")
        loss = state.get("latest_loss")

        if batch_idx % self.log_every_n_batches == 0:
            if not state.get('is_main_process', True):
                return
            
            batch_data = {
                "event": "batch",
                "epoch": state.get('current_epoch', 0),
                "batch": batch_idx,
                "loss": loss,
                "perplexity": self._calc_perplexity(loss)
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
    
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        epoch = state.get('current_epoch')
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