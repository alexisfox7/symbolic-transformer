# trainers/json_trainer.py
"""
Simplified integration helpers for JSON logging with lightweight metric checkpoints.
"""

import logging
import torch
import math
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from utils.json_logger import JSONLogger, create_json_logger_for_training


class JSONLoggingAccelerateTrainer:
    """
    Enhanced AccelerateTrainer with lightweight metric checkpoints every N batches.
    Simple approach - no hanging validation, just save metrics for later analysis.
    """
    
    def __init__(self, accelerate_trainer, json_logger=None, val_dataloader=None, 
                 checkpoint_every_n_batches=100, validate_every_n_batches=None, metrics_save_dir=None):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.val_dataloader = val_dataloader
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
        # Handle both parameter names for backward compatibility
        if validate_every_n_batches is not None:
            self.checkpoint_every_n_batches = validate_every_n_batches
        else:
            self.checkpoint_every_n_batches = checkpoint_every_n_batches
            
        self.logger = logging.getLogger(__name__)
        self.global_batch_count = 0
        
        # Setup metrics save directory
        self.metrics_save_dir = metrics_save_dir or os.path.join(
            getattr(accelerate_trainer, 'output_dir', './outputs'), 'batch_metrics'
        )
        if accelerate_trainer.accelerator.is_main_process:
            os.makedirs(self.metrics_save_dir, exist_ok=True)
        
        # Track accelerate-specific info
        self.accelerator = accelerate_trainer.accelerator
    
    def run_validation(self, epoch, batch_idx, global_batch):
        """
        Run validation and log results.
        """
        if self.val_dataloader is None:
            print("⚠️  No validation dataloader available")
            return {}
        
        print(f"🧪 Starting validation at batch {global_batch}...")
        
        # Set model to eval mode
        self.trainer.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        val_batches = 0
        
        with torch.no_grad():
            for val_batch_data in self.val_dataloader:
                # Forward pass
                outputs = self.trainer.model(**val_batch_data)
                loss = outputs.get('loss')
                
                if loss is not None and not torch.isnan(loss):
                    batch_size = val_batch_data.get('input_ids', next(iter(val_batch_data.values()))).size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    val_batches += 1
        
        # Set model back to train mode
        self.trainer.model.train()
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        val_metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'samples': total_samples,
            'batches': val_batches
        }
        
        print(f"📊 Validation results: loss={avg_loss:.4f}, ppl={perplexity:.2f}, samples={total_samples}")
        
        # Log to JSON (only on main process)
        if self.json_logger and self.accelerator.is_main_process:
            try:
                self.json_logger.log_validation(
                    epoch=epoch,
                    metrics={
                        'step': global_batch,
                        'val_loss': avg_loss,
                        'val_perplexity': perplexity,
                        'val_samples': total_samples,
                        'val_batches': val_batches
                    }
                )
                print(f"✅ Validation logged to JSON")
            except Exception as e:
                print(f"❌ Validation JSON logging failed: {e}")
        
        return val_metrics

    def save_batch_metrics(self, epoch, batch_idx, loss, model_state=None):
        """
        Save lightweight metrics checkpoint with minimal model info.
        Only saves what's needed to calculate perplexity and validation later.
        """
        if not self.accelerator.is_main_process:
            return
        
        # Calculate basic metrics
        train_perplexity = math.exp(loss) if loss < 20 else float('inf')
        
        # Create metrics snapshot
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'global_batch': self.global_batch_count,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'train_loss': loss,
            'train_perplexity': train_perplexity,
            'model_training_state': {
                'epoch': epoch,
                'global_step': self.global_batch_count,
                'loss': loss,
            }
        }
        
        # Save metrics only (not full model weights)
        metrics_file = os.path.join(
            self.metrics_save_dir, 
            f'metrics_batch_{self.global_batch_count:06d}.json'
        )
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Successfully saved batch metrics to {metrics_file}")
            
            # Optionally save minimal model checkpoint for validation later
            if model_state and self.val_dataloader:
                checkpoint_file = os.path.join(
                    self.metrics_save_dir,
                    f'model_batch_{self.global_batch_count:06d}.pt'
                )
                # Save minimal checkpoint - just model state dict
                torch.save({
                    'model_state_dict': self.accelerator.unwrap_model(self.trainer.model).state_dict(),
                    'global_batch': self.global_batch_count,
                    'epoch': epoch,
                    'train_loss': loss,
                }, checkpoint_file)
                
                self.logger.info(f"Successfully saved model checkpoint to {checkpoint_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to save batch metrics: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def train(self):
        """Training loop with lightweight metric checkpoints."""
        # Store original methods
        original_log_batch = self.trainer.log_batch
        original_log_epoch = self.trainer.log_epoch
        
        # Enhance batch logging
        def enhanced_log_batch(batch_idx, loss, epoch=None, metrics=None):
            # ALWAYS print this first - no process checks
            print(f"🔥 BATCH {batch_idx}: loss={loss:.4f}, epoch={epoch}")
            
            # Call original logging
            try:
                original_log_batch(batch_idx, loss, epoch=epoch, metrics=metrics)
            except Exception as e:
                print(f"⚠️  Original log_batch failed: {e}")
            
            # FIXED: Use global_batch from metrics (this is the actual global batch count)
            if metrics and 'global_batch' in metrics:
                self.global_batch_count = metrics['global_batch']
                print(f"🎯 Using global_batch from metrics: {self.global_batch_count}")
            else:
                self.global_batch_count += 1
                print(f"⚠️  No global_batch in metrics, incrementing: {self.global_batch_count}")

            # Print validation check info (use the actual global batch from accelerate trainer)
            current_global_batch = metrics.get('global_batch', self.global_batch_count) if metrics else self.global_batch_count
            remainder = current_global_batch % self.checkpoint_every_n_batches if self.checkpoint_every_n_batches > 0 else -1
            #print(f"📊 Global batch {current_global_batch}, validation check: {current_global_batch} % {self.checkpoint_every_n_batches} = {remainder}")
            
            # REPLACED: Run validation instead of saving checkpoints
            if self.checkpoint_every_n_batches > 0 and remainder == 0 and self.val_dataloader is not None:
                print(f"🧪 VALIDATION TRIGGER: Running validation at global batch {current_global_batch}")
                
                try:
                    # Run validation
                    val_metrics = self.run_validation(
                        epoch=epoch or 0,
                        batch_idx=batch_idx,
                        global_batch=current_global_batch
                    )
                    print(f"✅ Validation complete: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.2f}")
                    
                except Exception as e:
                    print(f"❌ Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # JSON logging (only on main process to avoid duplicate logs)
            if self.json_logger and self.accelerator.is_main_process:
                if current_global_batch % self.json_logger.log_every_n_steps == 0:
                    print(f"📝 Logging to JSON at batch {current_global_batch}")
                    
                    # Calculate train perplexity
                    train_perplexity = math.exp(loss) if loss < 20 else float('inf')
                    
                    batch_metrics = {
                        'loss': loss,
                        'perplexity': train_perplexity,
                        'global_batch': current_global_batch
                    }
                    if metrics:
                        batch_metrics.update(metrics)
                    
                    try:
                        self.json_logger.log_batch(
                            epoch=epoch or 0,
                            batch=batch_idx,
                            step=current_global_batch,
                            metrics=batch_metrics
                        )
                        print(f"✅ JSON logged successfully")
                    except Exception as e:
                        print(f"❌ JSON logging failed: {e}")
            
            # SIMPLE: Save lightweight checkpoint every N batches
            # Always show when we're close to checkpoints
            
            if self.checkpoint_every_n_batches > 0 and remainder == 0:
                #print(f"🎯 CHECKPOINT: Triggering save at global batch {self.global_batch_count}")
                if self.accelerator.is_main_process:
                    print(f"📁 Saving to: {self.metrics_save_dir}")
                else:
                    print(f"⚠️  Not main process, skipping save")
                    
                self.save_batch_metrics(
                    epoch=epoch or 0,
                    batch_idx=batch_idx,
                    loss=loss,
                    model_state=True  # Save model state for later validation
                )
        
        # Enhance epoch logging (keep simple - no mid-epoch validation)
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
                
                # Log epoch metrics
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
                    'total_batches': self.global_batch_count,
                    'metrics_save_dir': self.metrics_save_dir
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
    checkpoint_every_n_batches=100, validate_every_n_batches=None, metrics_save_dir=None, **trainer_kwargs
):
    """
    Create AccelerateTrainer with lightweight metric checkpoints.
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
        checkpoint_every_n_batches=checkpoint_every_n_batches,
        validate_every_n_batches=validate_every_n_batches,
        metrics_save_dir=metrics_save_dir
    )


def add_json_logging_args(parser):
    """Add JSON logging arguments to existing argument parser."""
    parser.add_argument("--json_log_steps", type=int, default=100,
                       help="Log training metrics every N batches to JSON (default: 100)")
    parser.add_argument("--checkpoint_every_n_batches", type=int, default=100,
                       help="Save lightweight checkpoint every N batches (default: 100)")
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


# Utility function to analyze saved metrics later
def analyze_batch_metrics(metrics_dir):
    """
    Analyze saved batch metrics to calculate validation and perplexity trends.
    Run this after training to get insights from the saved checkpoints.
    """
    import glob
    
    metrics_files = sorted(glob.glob(os.path.join(metrics_dir, 'metrics_batch_*.json')))
    
    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}")
        return
    
    print(f"Found {len(metrics_files)} metric checkpoints")
    
    # Load and analyze metrics
    batch_losses = []
    batch_perplexities = []
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            batch_losses.append(data['train_loss'])
            batch_perplexities.append(data['train_perplexity'])
            
            print(f"Batch {data['global_batch']:6d}: Loss={data['train_loss']:.4f}, PPL={data['train_perplexity']:.2f}")
            
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    if batch_losses:
        print(f"\nSummary:")
        print(f"  Average Loss: {sum(batch_losses)/len(batch_losses):.4f}")
        print(f"  Average Perplexity: {sum(batch_perplexities)/len(batch_perplexities):.2f}")
        print(f"  Final Loss: {batch_losses[-1]:.4f}")
        print(f"  Final Perplexity: {batch_perplexities[-1]:.2f}")