# trainers/json_trainer.py
"""
Simplified trainer integration for JSON logging.
"""

import logging
from typing import Optional, Dict, Any


class TrainerWithJSONLogging:
    """Simple wrapper to add JSON logging to any trainer."""
    
    def __init__(self, trainer, json_logger=None, log_interval=100):
        self.trainer = trainer
        self.json_logger = json_logger
        self.log_interval = log_interval
        self.step = 0
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """Run training with JSON logging."""
        # Log config if JSON logger is available
        if self.json_logger:
            config = {
                'num_epochs': self.trainer.num_epochs,
                'batch_size': self.trainer.dataloader.batch_size,
                'device': str(self.trainer.device),
                'model_params': sum(p.numel() for p in self.trainer.model.parameters())
            }
            self.json_logger.log_config(config)
        
        # Store original logging methods
        original_log_batch = self.trainer.log_batch
        original_log_epoch = self.trainer.log_epoch
        
        # Enhanced batch logging
        def log_batch_with_json(batch_idx, loss, epoch=None, metrics=None):
            # Call original logger
            original_log_batch(batch_idx, loss, epoch, metrics)
            
            # Increment step counter
            self.step += 1
            
            # Log to JSON if enabled and at interval
            if self.json_logger and self.step % self.log_interval == 0:
                json_metrics = {'loss': loss}
                if metrics:
                    json_metrics.update(metrics)
                self.json_logger.log_metrics(epoch or 0, self.step, json_metrics)
        
        # Enhanced epoch logging
        def log_epoch_with_json(epoch, avg_loss, metrics=None):
            # Call original logger
            original_log_epoch(epoch, avg_loss, metrics)
            
            # Log to JSON if enabled
            if self.json_logger:
                json_metrics = {'avg_loss': avg_loss}
                if metrics:
                    json_metrics.update(metrics)
                self.json_logger.log_epoch_summary(epoch, json_metrics)
        
        # Replace methods
        self.trainer.log_batch = log_batch_with_json
        self.trainer.log_epoch = log_epoch_with_json
        
        try:
            # Run training
            result = self.trainer.train()
            
            # Log completion
            if self.json_logger:
                self.json_logger.log_end(result if isinstance(result, dict) else {})
            
            return result
            
        finally:
            # Restore original methods
            self.trainer.log_batch = original_log_batch
            self.trainer.log_epoch = original_log_epoch
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_trainer_with_json_logging(trainer_class, json_logger=None, log_interval=100, **trainer_kwargs):
    """Create a trainer with JSON logging support."""
    base_trainer = trainer_class(**trainer_kwargs)
    return TrainerWithJSONLogging(base_trainer, json_logger, log_interval)