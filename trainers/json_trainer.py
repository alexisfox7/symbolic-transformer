# trainers/json_trainer.py
"""
Simplified integration helpers for JSON logging without gradient accumulation complexity.
"""

import logging
import torch
import math
from typing import Dict, Any, Optional
from utils.json_logger import JSONLogger, create_json_logger_for_training


class JSONLoggingAccelerateTrainer:
    """
    Enhanced AccelerateTrainer with proper distributed validation and perplexity logging.
    FIXED VERSION - uses accelerator.wait_for_everyone() for proper synchronization.
    """
    
    def __init__(self, accelerate_trainer, json_logger=None, val_dataloader=None, 
                 validate_every_n_batches=50):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.val_dataloader = val_dataloader
        self.validate_every_n_batches = validate_every_n_batches
        self.logger = logging.getLogger(__name__)
        self.global_batch_count = 0
        
        # Track accelerate-specific info
        self.accelerator = accelerate_trainer.accelerator
    
    def run_validation(self, model, val_dataloader, device):
        """
        PROPER distributed validation with accelerator synchronization.
        All processes participate and results are gathered correctly.
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        # Prepare validation dataloader if not already prepared
        if not hasattr(val_dataloader, '_accelerator_prepared'):
            val_dataloader = self.accelerator.prepare(val_dataloader)
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                # Data is already on correct device from accelerator.prepare()
                outputs = model(**batch_data)
                loss = outputs.get('loss')
                
                if loss is not None and not torch.isnan(loss):
                    batch_size = next(iter(batch_data.values())).size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
        
        # CRITICAL: Gather results from all processes using accelerator
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        
        # Use accelerator.gather() to collect from all processes
        all_losses = self.accelerator.gather(total_loss_tensor)
        all_samples = self.accelerator.gather(total_samples_tensor)
        
        # Sum across all processes (only on main process)
        if self.accelerator.is_main_process:
            total_loss = all_losses.sum().item()
            total_samples = all_samples.sum().item()
        else:
            # Non-main processes get dummy values
            total_loss = 0.0
            total_samples = 1
        
        model.train()
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        return {
            'loss': avg_loss, 
            'perplexity': perplexity, 
            'samples': total_samples
        }
    
    def train(self):
        """Training loop with proper distributed validation and perplexity logging."""
        # Store original methods
        original_log_batch = self.trainer.log_batch
        original_log_epoch = self.trainer.log_epoch
        
        # Enhance batch logging
        def enhanced_log_batch(batch_idx, loss, epoch=None, metrics=None):
            # Call original logging
            original_log_batch(batch_idx, loss, epoch=epoch, metrics=metrics)
            
            self.global_batch_count += 1
            
            # Only on main process for JSON logging
            if self.json_logger and self.accelerator.is_main_process:
                # Calculate train perplexity
                train_perplexity = math.exp(loss) if loss < 20 else float('inf')
                
                # Only log every N steps to JSON
                if self.global_batch_count % self.json_logger.log_every_n_steps == 0:
                    batch_metrics = {
                        'loss': loss,
                        'perplexity': train_perplexity,
                        'global_batch': self.global_batch_count
                    }
                    if metrics:
                        batch_metrics.update(metrics)
                    
                    self.json_logger.log_batch(
                        epoch=epoch or 0,
                        batch=batch_idx,
                        step=self.global_batch_count,
                        metrics=batch_metrics
                    )
            
            # FIXED: Proper mid-epoch validation with synchronization
            if (self.val_dataloader and 
                self.global_batch_count % self.validate_every_n_batches == 0):
                
                # CRITICAL: All processes must participate in validation
                # Use wait_for_everyone() to ensure all processes reach this point
                self.accelerator.wait_for_everyone()
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"Running validation at batch {self.global_batch_count}...")
                
                # All processes run validation (required for gather operations)
                val_metrics = self.run_validation(
                    self.trainer.model, 
                    self.val_dataloader, 
                    self.accelerator.device
                )
                
                # Only main process logs the results
                if self.accelerator.is_main_process:
                    self.logger.info(f"Batch {self.global_batch_count} Validation - Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
                    
                    if self.json_logger:
                        self.json_logger.log_validation(f"batch_{self.global_batch_count}", val_metrics)
                
                # Wait again to ensure all processes finish validation before continuing
                self.accelerator.wait_for_everyone()
        
        # Enhance epoch logging
        def enhanced_log_epoch(epoch: int, avg_loss: float, metrics=None):
            # Call original logging
            original_log_epoch(epoch, avg_loss, metrics)
            
            # Only on main process
            if self.json_logger and self.accelerator.is_main_process:
                # Calculate train perplexity
                train_perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                
                epoch_metrics = {
                    'loss': avg_loss,
                    'train_perplexity': train_perplexity
                }
                if metrics:
                    epoch_metrics.update(metrics)
                
                # Epoch-end validation with proper synchronization
                if self.val_dataloader:
                    # Ensure all processes participate
                    self.accelerator.wait_for_everyone()
                    
                    if self.accelerator.is_main_process:
                        self.logger.info(f"Running epoch {epoch} validation...")
                    
                    val_metrics = self.run_validation(
                        self.trainer.model,
                        self.val_dataloader,
                        self.accelerator.device
                    )
                    
                    if self.accelerator.is_main_process:
                        epoch_metrics.update({
                            'val_loss': val_metrics['loss'],
                            'val_perplexity': val_metrics['perplexity']
                        })
                        
                        self.logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.2f}, Val Loss: {val_metrics['loss']:.4f}, Val PPL: {val_metrics['perplexity']:.2f}")
                        
                        # Log validation separately too
                        self.json_logger.log_validation(epoch, val_metrics)
                        
                        # Log epoch metrics
                        self.json_logger.log_epoch_end(epoch, epoch_metrics)
                    
                    # Wait for all processes to complete validation
                    self.accelerator.wait_for_everyone()
                else:
                    # No validation case - still log epoch metrics on main process
                    if self.accelerator.is_main_process:
                        self.json_logger.log_epoch_end(epoch, epoch_metrics)
        
        # Replace trainer methods
        self.trainer.log_batch = enhanced_log_batch
        self.trainer.log_epoch = enhanced_log_epoch
        
        try:
            # Run training
            result = self.trainer.train()
            
            # Log final results (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                final_metrics = result.copy() if isinstance(result, dict) else {}
                final_metrics.update({
                    'total_processes': self.accelerator.num_processes,
                    'final_device': str(self.accelerator.device),
                    'total_batches': self.global_batch_count
                })
                self.json_logger.log_experiment_end(final_metrics)
            
            return result
            
        finally:
            # Restore original methods
            self.trainer.log_batch = original_log_batch
            self.trainer.log_epoch = original_log_epoch
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, device, json_logger=None, val_dataloader=None, 
    validate_every_n_batches=50, **trainer_kwargs
):
    """
    Create AccelerateTrainer with validation and perplexity logging.
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
    
    # Wrap with enhanced JSON logging
    return JSONLoggingAccelerateTrainer(
        trainer, 
        json_logger=json_logger,
        val_dataloader=val_dataloader,
        validate_every_n_batches=validate_every_n_batches
    )


def add_json_logging_args(parser):
    """Add JSON logging arguments to existing argument parser."""
    parser.add_argument("--json_log_steps", type=int, default=100,
                       help="Log training metrics every N batches to JSON (default: 100)")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    return parser


def create_json_logger_from_args(args, experiment_name="experiment"):
    """Create JSON logger from parsed arguments."""
    if getattr(args, 'disable_json_logging', False):
        return None
    
    log_steps = getattr(args, 'json_log_steps', 100)
    return create_json_logger_for_training(
        args.output_dir, 
        experiment_name, 
        log_steps
    )