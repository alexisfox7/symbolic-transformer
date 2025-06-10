# utils/simple_json_logger.py
"""
Simple, clean JSON logger without complexity.
Replaces the current wonky dual-logger setup.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional


class SimpleJSONLogger:
    """Clean, simple JSON logger for training metrics."""
    
    def __init__(self, output_dir: str, experiment_name: str, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.step_count = 0
        
        # Single log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.jsonl')
        
        # Write initial entry
        self._write_log({
            "event": "start",
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "log_file": self.log_file
        })
    
    def _write_log(self, data: Dict[str, Any]):
        """Write single JSON line to file."""
        data.update({
            "elapsed_time": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception:
            pass  # Silent fail - don't break training
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration once at start."""
        self._write_log({
            "event": "config", 
            "config": config
        })
    
    def log_step(self, epoch: int, batch: int, step: int, loss: float, **extra_metrics):
        """Log training step if at interval."""
        self.step_count = step
        if step % self.log_every_n_steps == 0:
            self._write_log({
                "event": "step",
                "epoch": epoch,
                "batch": batch, 
                "step": step,
                "loss": loss,
                **extra_metrics
            })
    
    def log_epoch(self, epoch: int, avg_loss: float, **extra_metrics):
        """Log epoch completion."""
        self._write_log({
            "event": "epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            **extra_metrics
        })
    
    def log_validation(self, epoch: int, val_loss: float, **extra_metrics):
        """Log validation results."""
        self._write_log({
            "event": "validation",
            "epoch": epoch,
            "val_loss": val_loss,
            **extra_metrics
        })


def create_simple_json_logger(output_dir: str, experiment_name: str, log_steps: int = 100) -> Optional[SimpleJSONLogger]:
    """Create simple JSON logger."""
    return SimpleJSONLogger(output_dir, experiment_name, log_steps)


# Simple trainer wrapper that adds JSON logging without complexity
class SimpleJSONTrainerWrapper:
    """Simple wrapper that adds JSON logging to any trainer."""
    
    def __init__(self, trainer, json_logger: Optional[SimpleJSONLogger] = None):
        self.trainer = trainer
        self.json_logger = json_logger
        self.step_count = 0
    
    def train(self):
        """Train with simple JSON logging."""
        if self.json_logger:
            # Log basic config once
            config = {
                'num_epochs': getattr(self.trainer, 'num_epochs', 'unknown'),
                'batch_size': getattr(self.trainer.dataloader, 'batch_size', 'unknown') if hasattr(self.trainer, 'dataloader') else 'unknown'
            }
            # Add accelerate info if available
            if hasattr(self.trainer, 'accelerator'):
                config.update({
                    'num_processes': self.trainer.accelerator.num_processes,
                    'device': str(self.trainer.accelerator.device)
                })
            self.json_logger.log_config(config)
        
        # Wrap original logging methods simply
        original_log_batch = getattr(self.trainer, 'log_batch', None)
        original_log_epoch = getattr(self.trainer, 'log_epoch', None)
        
        if original_log_batch and self.json_logger:
            def new_log_batch(batch_idx, loss, epoch=None, metrics=None):
                original_log_batch(batch_idx, loss, epoch, metrics)
                # Only log on main process for distributed training
                if not hasattr(self.trainer, 'accelerator') or self.trainer.accelerator.is_main_process:
                    self.step_count += 1
                    extra = metrics or {}
                    self.json_logger.log_step(epoch or 0, batch_idx, self.step_count, loss, **extra)
            self.trainer.log_batch = new_log_batch
        
        if original_log_epoch and self.json_logger:
            def new_log_epoch(epoch, avg_loss, metrics=None):
                original_log_epoch(epoch, avg_loss, metrics)
                # Only log on main process for distributed training
                if not hasattr(self.trainer, 'accelerator') or self.trainer.accelerator.is_main_process:
                    extra = metrics or {}
                    self.json_logger.log_epoch(epoch, avg_loss, **extra)
            self.trainer.log_epoch = new_log_epoch
        
        # Run training
        try:
            return self.trainer.train()
        finally:
            # Restore original methods
            if original_log_batch:
                self.trainer.log_batch = original_log_batch
            if original_log_epoch:
                self.trainer.log_epoch = original_log_epoch
    
    def __getattr__(self, name):
        """Delegate everything else to wrapped trainer."""
        return getattr(self.trainer, name)