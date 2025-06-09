# trainers/json_trainer.py
"""
Fixed integration helpers for JSON logging with proper step-based tracking.
Correctly handles gradient accumulation for AccelerateTrainer.
"""

import logging
from typing import Dict, Any, Optional
from utils.json_logger import JSONLogger, create_json_logger_for_training


class StepTrackingAccelerateTrainer:
    """
    Fixed wrapper around AccelerateTrainer for proper step-based JSON logging.
    Correctly tracks gradient update steps, not mini-batch steps.
    """
    
    def __init__(self, accelerate_trainer, json_logger: Optional[JSONLogger] = None):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.logger = logging.getLogger(__name__)
        
        # Track accelerate-specific info
        self.accelerator = accelerate_trainer.accelerator
        self.gradient_accumulation_steps = accelerate_trainer.gradient_accumulation_steps
    
    def train(self):
        """Training loop with corrected step-based JSON logging for AccelerateTrainer."""
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
        
        # Enhance the log_batch method to properly track gradient update steps
        original_log_batch = self.trainer.log_batch
        
        def enhanced_log_batch(step_idx: int, loss: float, epoch: Optional[int] = None, metrics: Optional[Dict[str, Any]] = None):
            # Call original logging
            original_log_batch(step_idx, loss, epoch, metrics)
            
            # Add JSON logging (only on main process)
            # step_idx is now correctly the gradient update step from the fixed trainer
            if self.json_logger and self.accelerator.is_main_process:
                step_metrics = {'loss': loss}
                if metrics:
                    step_metrics.update(metrics)
                
                # Add accelerate-specific metrics
                step_metrics.update({
                    'is_gradient_update_step': True,
                    'num_processes': self.accelerator.num_processes
                })
                
                self.json_logger.log_step(
                    step=step_idx,  # This is now correctly the gradient update step
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
                epoch_metrics = {'loss': avg_loss}
                if metrics:
                    epoch_metrics.update(metrics)
                
                # Add accelerate-specific info
                epoch_metrics.update({
                    'num_processes': self.accelerator.num_processes,
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
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
                    'epoch': epoch,
                    'num_processes': self.accelerator.num_processes
                })
                self.json_logger.log_checkpoint(epoch or 0, path, checkpoint_metrics)
        
        self.trainer.save_checkpoint = enhanced_save_checkpoint
        
        # Enhance batch-end callback to track steps correctly
        original_trigger_callbacks = self.trainer._trigger_callbacks
        
        def enhanced_trigger_callbacks(event_name: str, *args, **kwargs):
            # Call original callback triggering
            original_trigger_callbacks(event_name, *args, **kwargs)
            
            # Add JSON logging for batch ends with correct step tracking
            if (event_name == 'on_batch_end' and 
                self.json_logger and 
                self.accelerator.is_main_process and 
                len(args) >= 2):
                
                batch_idx = args[0]
                logs = args[1] if len(args) > 1 else kwargs.get('logs', {})
                
                if isinstance(logs, dict) and 'global_step' in logs:
                    global_step = logs['global_step']
                    loss = logs.get('loss')
                    
                    # Only log if this was a gradient update step and we have a valid loss
                    # Check if this batch resulted in a gradient update
                    if ((batch_idx + 1) % self.gradient_accumulation_steps == 0 and 
                        loss is not None and 
                        global_step % self.json_logger.log_every_n_steps == 0):
                        
                        self.json_logger.log_batch(
                            epoch=self.trainer.trainer_state.get('current_epoch', 0),
                            batch=batch_idx + 1,
                            step=global_step,
                            metrics={'loss': loss, 'is_gradient_update': True}
                        )
        
        self.trainer._trigger_callbacks = enhanced_trigger_callbacks
        
        # Run the actual training
        try:
            result = self.trainer.train()
            
            # Log final results (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                final_metrics = result.copy() if isinstance(result, dict) else {}
                final_metrics.update({
                    'total_processes': self.accelerator.num_processes,
                    'final_device': str(self.accelerator.device),
                    'gradient_accumulation_steps': self.gradient_accumulation_steps
                })
                self.json_logger.log_experiment_end(final_metrics)
            
            return result
            
        except Exception as e:
            if self.json_logger and self.accelerator.is_main_process:
                self.json_logger.log_custom("training_error", {
                    "error": str(e),
                    "process_index": self.accelerator.process_index,
                    "num_processes": self.accelerator.num_processes
                })
            raise
        
        finally:
            # Restore original methods
            self.trainer.log_batch = original_log_batch
            self.trainer.log_epoch = original_log_epoch
            self.trainer.save_checkpoint = original_save_checkpoint
            self.trainer._trigger_callbacks = original_trigger_callbacks
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, device, json_logger=None, **trainer_kwargs
):
    """
    Create AccelerateTrainer with corrected JSON logging integration.
    
    Args:
        model: Model to train
        dataloader: Training dataloader  
        optimizer: Optimizer
        device: Device (will be overridden by accelerator)
        json_logger: Optional JSONLogger instance
        **trainer_kwargs: Additional arguments for AccelerateTrainer
        
    Returns:
        StepTrackingAccelerateTrainer instance with corrected step tracking
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
    
    # Wrap with corrected JSON logging
    return StepTrackingAccelerateTrainer(trainer, json_logger)


# Quick CLI argument helper
def add_json_logging_args(parser):
    """Add JSON logging arguments to existing argument parser."""
    parser.add_argument("--json_log_steps", type=int, default=256,
                       help="Log training metrics every N gradient update steps to JSON (default: 256)")
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