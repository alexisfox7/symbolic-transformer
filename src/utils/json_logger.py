# utils/json_logger.py
"""
Simple JSON logging utility for training metrics and events.
Provides structured logging without overcomplicating the existing system.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class JSONLogger:
    """Simple JSON logger for training metrics and events."""
    
    def __init__(self, log_file: str, experiment_name: str = None, log_every_n_steps: int = 100):
        """
        Initialize JSON logger.
        
        Args:
            log_file: Path to JSON log file
            experiment_name: Optional experiment identifier
            log_every_n_steps: Log training metrics every N steps (default: 100)
        """
        self.log_file = log_file
        self.experiment_name = experiment_name or "experiment"
        self.log_every_n_steps = log_every_n_steps
        self.start_time = time.time()
        self.step_count = 0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log file with experiment metadata
        self._log_event("experiment_start", {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "log_file": self.log_file,
            "log_every_n_steps": self.log_every_n_steps
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Write a JSON event to the log file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "event_type": event_type,
            "experiment_name": self.experiment_name,
            **data
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.warning(f"Failed to write JSON log: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration/hyperparameters."""
        self._log_event("config", {"config": config})
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start."""
        self._log_event("epoch_start", {"epoch": epoch})
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch completion with metrics."""
        self._log_event("epoch_end", {
            "epoch": epoch,
            "metrics": metrics
        })
    
    def log_step(self, step: int, epoch: int, metrics: Dict[str, Any]):
        """Log training step metrics."""
        self.step_count = step
        if step % self.log_every_n_steps == 0:
            self._log_event("training_step", {
                "step": step,
                "epoch": epoch,
                "metrics": metrics
            })
    
    def log_batch(self, epoch: int, batch: int, step: int, metrics: Dict[str, Any]):
        """Log batch metrics (only at step intervals)."""
        self.step_count = step
        if step % self.log_every_n_steps == 0:
            self._log_event("batch", {
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "metrics": metrics
            })
    
    def log_validation(self, epoch: int, metrics: Dict[str, Any]):
        """Log validation metrics."""
        self._log_event("validation", {
            "epoch": epoch,
            "metrics": metrics
        })
    
    def log_generation(self, epoch: int, prompt: str, generated: str, 
                      generation_params: Dict[str, Any] = None):
        """Log text generation examples."""
        self._log_event("generation", {
            "epoch": epoch,
            "prompt": prompt,
            "generated": generated,
            "generation_params": generation_params or {}
        })
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, metrics: Dict[str, Any] = None):
        """Log checkpoint save."""
        self._log_event("checkpoint", {
            "epoch": epoch,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics or {}
        })
    
    def log_experiment_end(self, final_metrics: Dict[str, Any] = None):
        """Log experiment completion."""
        self._log_event("experiment_end", {
            "end_time": datetime.now().isoformat(),
            "total_elapsed_time": time.time() - self.start_time,
            "final_metrics": final_metrics or {}
        })
    
    def log_custom(self, event_name: str, data: Dict[str, Any]):
        """Log custom event."""
        self._log_event(event_name, data)


class JSONLoggerCallback:
    """Callback for integrating JSONLogger with existing trainer system."""
    
    def __init__(self, json_logger: JSONLogger):
        self.json_logger = json_logger
        self.current_step = 0
        self.current_epoch = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at training start."""
        if logs:
            self.json_logger.log_config(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at epoch start."""
        self.current_epoch = epoch
        self.json_logger.log_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at epoch end."""
        if logs:
            self.json_logger.log_epoch_end(epoch, logs)
    
    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Called at batch end - logs every N steps."""
        if logs and 'loss' in logs and logs['loss'] is not None:
            self.current_step += 1
            self.json_logger.log_step(
                step=self.current_step,
                epoch=self.current_epoch, 
                metrics={'loss': logs['loss'], 'batch': batch_idx}
            )
    
    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at evaluation end."""
        if logs:
            epoch = logs.get('epoch', self.current_epoch)
            self.json_logger.log_validation(epoch, logs)


def load_json_logs(log_file: str) -> list:
    """
    Load and parse JSON log file.
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        List of parsed JSON events
    """
    events = []
    if not os.path.exists(log_file):
        return events
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse JSON line: {e}")
    except Exception as e:
        logging.error(f"Failed to read JSON log file: {e}")
    
    return events


def extract_metrics_from_logs(log_file: str) -> Dict[str, list]:
    """
    Extract training metrics from JSON logs.
    
    Args:
        log_file: Path to JSON log file
        
    Returns:
        Dictionary with metric lists (epochs, losses, etc.)
    """
    events = load_json_logs(log_file)
    
    metrics = {
        'epochs': [],
        'epoch_losses': [],
        'batch_losses': [],
        'validation_metrics': [],
        'generation_examples': []
    }
    
    for event in events:
        event_type = event.get('event_type')
        
        if event_type == 'epoch_end':
            epoch = event.get('epoch', 0)
            event_metrics = event.get('metrics', {})
            
            metrics['epochs'].append(epoch)
            if 'loss' in event_metrics:
                metrics['epoch_losses'].append(event_metrics['loss'])
        
        elif event_type == 'batch':
            batch_metrics = event.get('metrics', {})
            if 'loss' in batch_metrics:
                metrics['batch_losses'].append(batch_metrics['loss'])
        
        elif event_type == 'validation':
            metrics['validation_metrics'].append(event.get('metrics', {}))
        
        elif event_type == 'generation':
            metrics['generation_examples'].append({
                'epoch': event.get('epoch'),
                'prompt': event.get('prompt'),
                'generated': event.get('generated')
            })
    
    return metrics


# Usage example for integration with existing code
def create_json_logger_for_training(output_dir: str, experiment_name: str, log_every_n_steps: int = 100) -> JSONLogger:
    """
    Create JSON logger for training runs.
    
    Args:
        output_dir: Training output directory
        experiment_name: Name for this experiment
        log_every_n_steps: Log training metrics every N steps (default: 100)
        
    Returns:
        Configured JSONLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, 'logs', f'{experiment_name}_{timestamp}.jsonl')
    return JSONLogger(log_file, experiment_name, log_every_n_steps)