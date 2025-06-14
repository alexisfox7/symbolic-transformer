# trainers/json_trainer.py
"""
Simplified integration helpers for JSON logging with lightweight metric checkpoints.
FIXED VERSION - Debug statements will actually print and checkpoints will save.
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
    FIXED: Removed problematic main process checks and simplified logic.
    """
    
    def __init__(self, accelerate_trainer, json_logger=None, val_dataloader=None, 
                 checkpoint_every_n_batches=100, validate_every_n_batches=None, metrics_save_dir=None):
        self.trainer = accelerate_trainer
        self.json_logger = json_logger
        self.val_dataloader = val_dataloader
        
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
        
        # Create directory on all processes (avoid race conditions)
        os.makedirs(self.metrics_save_dir, exist_ok=True)
        
        # Track accelerate-specific info
        self.accelerator = accelerate_trainer.accelerator
        
        # FIXED: Print setup info immediately
        print(f"🚀 JSONLoggingAccelerateTrainer initialized:")
        print(f"  📋 Checkpoint every: {self.checkpoint_every_n_batches} batches")
        print(f"  📋 JSON log every: {self.json_logger.log_every_n_steps if self.json_logger else 'N/A'} steps")
        print(f"  📋 Process rank: {self.accelerator.process_index}/{self.accelerator.num_processes}")
        print(f"  📋 Main process: {self.accelerator.is_main_process}")
        print(f"  📋 Metrics dir: {self.metrics_save_dir}")
    
    def save_batch_metrics(self, epoch, batch_idx, loss, model_state=None):
        """
        Save lightweight metrics checkpoint with minimal model info.
        FIXED: Only save on main process but print on all processes.
        """
        print(f"💾 save_batch_metrics called: epoch={epoch}, batch={batch_idx}, loss={loss:.4f}")
        
        if not self.accelerator.is_main_process:
            print(f"⚠️  Not main process (rank {self.accelerator.process_index}), skipping file save")
            return
        
        try:
            # Calculate metrics
            train_perplexity = math.exp(loss) if loss < 20 else float('inf')
            
            # Create metrics data
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'global_batch': self.global_batch_count,
                'train_loss': loss,
                'train_perplexity': train_perplexity,
                'process_info': {
                    'rank': self.accelerator.process_index,
                    'num_processes': self.accelerator.num_processes,
                    'device': str(self.accelerator.device)
                }
            }
            
            # Save to file using the ACTUAL global batch count
            metrics_file = os.path.join(
                self.metrics_save_dir, 
                f'metrics_batch_{self.global_batch_count:06d}.json'
            )
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            print(f"✅ Metrics saved to: {metrics_file}")
            
            # Optional: Save model state if requested
            if model_state and hasattr(self.trainer, 'model'):
                model_file = os.path.join(
                    self.metrics_save_dir,
                    f'model_batch_{self.global_batch_count:06d}.pt'
                )
                torch.save(self.trainer.model.state_dict(), model_file)
                print(f"✅ Model state saved to: {model_file}")
                
        except Exception as e:
            print(f"❌ Error saving batch metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def train(self):
        """
        FIXED: Enhanced training with proper debug output and checkpoint saving.
        """
        print(f"🚀 Starting enhanced training with JSON logging")
        
        # Store original methods
        original_log_batch = self.trainer.log_batch
        original_log_epoch = self.trainer.log_epoch
        
        # FIXED: Enhanced batch logging with proper debug output
        def enhanced_log_batch(batch_idx, loss, epoch=None, metrics=None):
            # ALWAYS print this first - no process checks
            #print(f"🔥 BATCH {batch_idx}: loss={loss:.4f}, epoch={epoch}")
            
            # Call original logging
            try:
                original_log_batch(batch_idx, loss, epoch=epoch, metrics=metrics)
            except Exception as e:
                print(f"⚠️  Original log_batch failed: {e}")
            
            # FIXED: Use global_batch from metrics (this is the actual global batch count)
            if metrics and 'global_batch' in metrics:
                self.global_batch_count = metrics['global_batch']
                #print(f"🎯 Using global_batch from metrics: {self.global_batch_count}")
            else:
                self.global_batch_count += 1
                print(f"⚠️  No global_batch in metrics, incrementing: {self.global_batch_count}")
            
            # Print checkpoint check info (use the actual global batch from accelerate trainer)
            current_global_batch = metrics.get('global_batch', self.global_batch_count) if metrics else self.global_batch_count
            remainder = current_global_batch % self.checkpoint_every_n_batches if self.checkpoint_every_n_batches > 0 else -1
            #print(f"📊 Global batch {current_global_batch}, checkpoint check: {current_global_batch} % {self.checkpoint_every_n_batches} = {remainder}")
            
            # FIXED: Use the actual global batch count for checkpoint logic
            if self.checkpoint_every_n_batches > 0 and remainder == 0:
                #print(f"🎯 CHECKPOINT TRIGGER: Saving at global batch {current_global_batch}")
                self.save_batch_metrics(
                    epoch=epoch or 0,
                    batch_idx=batch_idx,
                    loss=loss,
                    model_state=True
                )
            
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
        
        # Enhanced epoch logging
        def enhanced_log_epoch(epoch: int, avg_loss: float, metrics=None):
            print(f"🏁 EPOCH {epoch} COMPLETE: avg_loss={avg_loss:.4f}")
            
            # Call original logging
            try:
                original_log_epoch(epoch, avg_loss, metrics)
            except Exception as e:
                print(f"⚠️  Original log_epoch failed: {e}")
            
            # Only on main process for JSON logging
            if self.json_logger and self.accelerator.is_main_process:
                # Calculate train perplexity
                train_perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                
                epoch_metrics = {
                    'loss': avg_loss,
                    'train_perplexity': train_perplexity
                }
                if metrics:
                    epoch_metrics.update(metrics)
                
                try:
                    self.json_logger.log_epoch_end(epoch, epoch_metrics)
                    print(f"✅ Epoch {epoch} logged to JSON")
                except Exception as e:
                    print(f"❌ Epoch JSON logging failed: {e}")
        
        # FIXED: Replace trainer methods with debug wrapper
        print(f"🔧 Replacing trainer methods...")
        self.trainer.log_batch = enhanced_log_batch
        self.trainer.log_epoch = enhanced_log_epoch
        print(f"✅ Methods replaced successfully")
        
        try:
            # Run training
            print(f"🏃 Starting actual training...")
            result = self.trainer.train()
            print(f"🏆 Training completed!")
            
            # Log final results (only on main process)
            if self.json_logger and self.accelerator.is_main_process:
                final_metrics = result.copy() if isinstance(result, dict) else {}
                final_metrics.update({
                    'total_processes': self.accelerator.num_processes,
                    'final_device': str(self.accelerator.device),
                    'total_batches': self.global_batch_count,
                    'metrics_save_dir': self.metrics_save_dir
                })
                
                try:
                    self.json_logger.log_experiment_end(final_metrics)
                    print(f"✅ Experiment end logged to JSON")
                except Exception as e:
                    print(f"❌ Experiment end logging failed: {e}")
            
            return result
            
        except Exception as e:
            print(f"💥 Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            # Restore original methods
            print(f"🔧 Restoring original trainer methods...")
            self.trainer.log_batch = original_log_batch
            self.trainer.log_epoch = original_log_epoch
            print(f"✅ Methods restored")
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped trainer."""
        return getattr(self.trainer, name)


def create_accelerate_trainer_with_json_logging(
    model, dataloader, optimizer, device, json_logger=None, val_dataloader=None, 
    checkpoint_every_n_batches=100, validate_every_n_batches=None, metrics_save_dir=None, **trainer_kwargs
):
    """
    Create AccelerateTrainer with lightweight metric checkpoints.
    FIXED: Better error handling and debug output.
    """
    from trainers import get_trainer
    
    print(f"🏗️  Creating AccelerateTrainer...")
    
    # Create base AccelerateTrainer
    trainer = get_trainer(
        trainer_type="accelerate",
        model=model,
        dataloader=dataloader, 
        optimizer=optimizer,
        device=device,
        **trainer_kwargs
    )
    
    print(f"✅ Base trainer created: {type(trainer)}")
    
    # Wrap with enhanced JSON logging
    wrapped_trainer = JSONLoggingAccelerateTrainer(
        trainer, 
        json_logger=json_logger,
        val_dataloader=val_dataloader,
        checkpoint_every_n_batches=checkpoint_every_n_batches,
        validate_every_n_batches=validate_every_n_batches,
        metrics_save_dir=metrics_save_dir
    )
    
    print(f"✅ Wrapped trainer created: {type(wrapped_trainer)}")
    return wrapped_trainer


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