# utils/json_logger.py
"""
Simplified JSON logging for training metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONLogger:
    """Simple JSON logger for training metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """Initialize JSON logger with minimal configuration."""
        self.log_dir = Path(log_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.experiment_name = experiment_name
        
        # Write initial metadata
        self._write({
            "event": "start",
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def _write(self, data: Dict[str, Any]):
        """Write a single JSON line to the log file."""
        with open(self.log_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration at start of training."""
        self._write({
            "event": "config",
            "timestamp": datetime.now().isoformat(),
            "config": config
        })
    
    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, Any]):
        """Log training metrics at a given step."""
        self._write({
            "event": "metrics",
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics
        })
    
    def log_epoch_summary(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch summary metrics."""
        self._write({
            "event": "epoch_summary",
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": metrics
        })
    
    def log_validation(self, epoch: int, metrics: Dict[str, Any]):
        """Log validation metrics."""
        self._write({
            "event": "validation",
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": metrics
        })
    
    def log_generation(self, epoch: int, examples: list):
        """Log text generation examples."""
        self._write({
            "event": "generation",
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "examples": examples
        })
    
    def log_end(self, final_metrics: Optional[Dict[str, Any]] = None):
        """Log training completion."""
        self._write({
            "event": "end",
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "metrics": final_metrics or {}
        })


def create_json_logger(output_dir: str, experiment_name: str) -> JSONLogger:
    """Create a JSON logger instance."""
    return JSONLogger(output_dir, experiment_name)