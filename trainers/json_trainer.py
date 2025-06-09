# trainers/json_trainer.py
"""
Integration helpers for JSON logging with step-based tracking.
Specifically designed for AccelerateTrainer integration.
"""

import logging
from typing import Dict, Any, Optional
from utils.json_logger import JSONLogger, create_json_logger_for_training


class StepTrackingAccelerateTrainer:
    """
    Wrapper around AccelerateTrainer to add step-based JSON logging.
    Integrates with the accelerate training loop and gradient accumulation.
    """
    
    def __init__(self, accelerate_trainer, json_logger: Optional[JSONLogger] = None):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.global_step = 0
        self.logger = logging.getLogger(__name__)
        
        # Track accelerate-specific info
        self.accelerator = accelerate_trainer.accelerator
        self.gradient_accumulation_steps = accelerate_trainer.gradient_accumulation_steps
    
    def train(self):
        """Training loop with step-based JSON logging for AccelerateTrainer."""
        if self.json_logger and self.accelerator.is_main_process:
            # Log initial config including accelerate info
            config_data = {
                'num_epochs': self.trainer.num_epochs,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'num_processes': self.accelerator.num_processes,
                'mixed_precision': str(self.accelerator.mixed_precision),
                'device': str(self.accelerator.device),
                'log_interval': getattr(self.trainer, 'log_interval', 10)
            }
            self.json_logger.log_config(config_data)
        
        # Enhance the original log_batch method for accelerate
        original_log_batch = self.trainer.log_batch
        
        def enhanced_log_batch(step_idx: int, loss: float, epoch: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None):
            # Call original logging
            original_log_batch(step_idx, loss, epoch, metrics)
            
            # Add JSON logging (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                # In accelerate trainer, step_idx is already the gradient update step
                self.global_step = step_idx
                
                step_metrics = {'loss': loss}
                if metrics:
                    step_metrics.update(metrics)
                
                # Add accelerate-specific metrics
                step_metrics.update({
                    'gradient_accumulation_step': step_idx % self.gradient_accumulation_steps,
                    'is_gradient_update': True  # This is called after gradient updates
                })
                
                self.json_logger.log_step(
                    step=self.global_step,
                    epoch=epoch or 0,
                    metrics=step_metrics
                )
        
        # Replace the trainer's log_batch method
        self.trainer.log_batch = enhanced_log_batch
        
        # Enhance epoch logging
        original_log_epoch = self.trainer.log_epoch
        
        def enhanced_log_epoch(epoch: int, avg_loss: float, metrics: Optional[Dict[str, Any]] = None):
            # Call original logging
            original_log_epoch(epoch, avg_loss, metrics)
            
            # Add JSON logging (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                epoch_metrics = {'loss': avg_loss, 'global_step': self.global_step}
                if metrics:
                    epoch_metrics.update(metrics)
                
                # Add accelerate-specific info
                epoch_metrics.update({
                    'num_processes': self.accelerator.num_processes,
                    'effective_batch_size': self.trainer.dataloader.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
                })
                
                self.json_logger.log_epoch_end(epoch, epoch_metrics)
        
        self.trainer.log_epoch = enhanced_log_epoch
        
        # Enhance checkpoint saving to include JSON logging
        original_save_checkpoint = self.trainer.save_checkpoint
        
        def enhanced_save_checkpoint(path: str, epoch: Optional[int] = None, **kwargs):
            # Call original checkpoint save
            original_save_checkpoint(path, epoch, **kwargs)
            
            # Add JSON logging (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                checkpoint_metrics = kwargs.copy()
                checkpoint_metrics.update({
                    'global_step': self.global_step,
                    'epoch': epoch
                })
                self.json_logger.log_checkpoint(epoch or 0, path, checkpoint_metrics)
        
        self.trainer.save_checkpoint = enhanced_save_checkpoint
        
        # Run the actual training
        try:
            result = self.trainer.train()
            
            # Log final results (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                final_metrics = result.copy() if isinstance(result, dict) else {}
                final_metrics.update({
                    'total_steps': self.global_step,
                    'total_processes': self.accelerator.num_processes,
                    'final_device': str(self.accelerator.device)
                })
                self.json_logger.log_experiment_end(final_metrics)
            
            return result
            
        except Exception as e:
            if self.json_logger and self.accelerator.is_main_process:
                self.json_logger.log_custom("training_error", {
                    "error": str(e),
                    "step": self.global_step,
                    "process_index": self.accelerator.process_index
                })
            raise
        
        finally:
            # Restore original methods
            self.trainer.log_batch = original_log_batch
            self.trainer.log_epoch = original_log_epoch
            self.trainer.save_checkpoint = original_save_checkpoint
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, device, json_logger=None, **trainer_kwargs
):
    """
    Create AccelerateTrainer with JSON logging integration.
    
    Args:
        model: Model to train
        dataloader: Training dataloader  
        optimizer: Optimizer
        device: Device (will be overridden by accelerator)
        json_logger: Optional JSONLogger instance
        **trainer_kwargs: Additional arguments for AccelerateTrainer
        
    Returns:
        StepTrackingAccelerateTrainer instance
    """
    from trainers import get_trainer
    
    # Create base AccelerateTrainer
    trainer = get_trainer(
        trainer_type="accelerate",
        model=model,
        dataloader=dataloader, 
        optimizer=optimizer,
        device=device,
        **trainer_kwargs
    )
    
    # Wrap with JSON logging
    return StepTrackingAccelerateTrainer(trainer, json_logger)


def add_json_logging_to_accelerate_training():
    """
    Example of how to add JSON logging to AccelerateTrainer.
    
    Replace:
        trainer = get_trainer("accelerate", ...)
        result = trainer.train()
    
    With:
        json_logger = create_json_logger_for_training(output_dir, "experiment", log_every_n_steps=256)
        trainer = create_accelerate_trainer_with_json_logging(
            model, dataloader, optimizer, device, json_logger, **kwargs
        )
        result = trainer.train()
    """
    pass


def add_json_logging_to_training_script():
    """
    Example of how to add JSON logging to existing training scripts with minimal changes.
    
    Just replace:
        trainer = get_trainer(...)
        result = trainer.train()
    
    With:
        trainer = get_trainer(...)
        json_logger = create_json_logger_for_training(output_dir, "experiment", log_every_n_steps=256)
        wrapped_trainer = StepTrackingSimpleTrainer(trainer, json_logger)
        result = wrapped_trainer.train()
    """
    pass

# Quick CLI argument helper
def add_json_logging_args(parser):
    """Add JSON logging arguments to existing argument parser."""
    parser.add_argument("--json_log_steps", type=int, default=256,
                       help="Log training metrics every N steps to JSON (default: 256)")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    return parser


def create_json_logger_from_args(args, experiment_name="experiment"):
    """Create JSON logger from parsed arguments."""
    if getattr(args, 'disable_json_logging', False):
        return None
    
    log_steps = getattr(args, 'json_log_steps', 256)
    return create_json_logger_for_training(
        args.output_dir, 
        experiment_name, 
        log_steps
    )