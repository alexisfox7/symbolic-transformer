#./trainers/accelerate_trainer.py
"""
Simplified Accelerate Trainer without gradient accumulation - FIXED VERSION.
Removes the hanging wait_for_everyone() call that causes terminal freezing.
Updated to use the hook system instead of callbacks.
"""

import time
import warnings
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Suppress accelerate kernel version warnings
warnings.filterwarnings("ignore", message=".*kernel version.*")
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

from accelerate import Accelerator
from accelerate.logging import get_logger

from .base_trainer import BaseTrainer

logger = get_logger(__name__)

class AccelerateTrainer(BaseTrainer):
    """Accelerate trainer without gradient accumulation complexity - FIXED with hooks."""

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 start_epoch=1):
        
        # Initialize accelerator (no gradient accumulation)
        self.accelerator = Accelerator(
            log_with=None,
            project_dir=output_dir
        )
        
        # Use accelerator's device instead of passed device
        super().__init__(model, dataloader, optimizer, self.accelerator.device, output_dir)
        
        # Add accelerator-specific items to trainer_state for hooks
        self.trainer_state['accelerator'] = self.accelerator
        self.trainer_state['is_main_process'] = self.accelerator.is_main_process

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
        

    def train(self) -> Dict[str, Any]:
        """Execute training loop with accelerate."""
        self.trainer_state['num_epochs'] = self.num_epochs
        self.hooks.call_hooks('on_train_begin', self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'),
            'training_time': 0.0,
            'total_batches': 0,
            'total_samples': 0
        }

        # Track total batches processed
        global_batch = 0

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            self.hooks.call_hooks('on_epoch_begin', self.trainer_state)
            epoch_start_time = time.time()
            
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
                disable=not self.accelerator.is_local_main_process
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                self.trainer_state['current_batch'] = batch_data
                self.hooks.call_hooks('on_batch_begin', self.trainer_state)
                
                # Forward pass
                outputs = self.model(**batch_data)
                total_loss, loss_dict = self.calculate_loss(outputs.get('loss'))

                if total_loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                    continue
                    
                if torch.isnan(total_loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                    self.trainer_state['status'] = 'NaN Loss'
                    self.hooks.call_hooks('on_train_end', self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                # Backward pass - accelerator handles gradient scaling
                self.accelerator.backward(total_loss)

                # Gradient clipping if specified
                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Track metrics
                batch_loss_item = total_loss.item()
                epoch_loss += batch_loss_item
                num_batches += 1
                global_batch += 1

                # Update progress bar
                progress_bar.set_postfix(loss_dict)
    
                # Calculate batch metrics for every batch
                batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).shape[0]
       
                self.trainer_state.update({
                    'latest_loss': batch_loss_item,
                    'global_batch': global_batch,
                    'current_batch_idx': batch_idx
                })
                self.hooks.call_hooks('on_batch_end', self.trainer_state)

            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['total_batches'] += num_batches
            training_metrics['total_samples'] += len(self.dataloader.dataset)

            epoch_duration = time.time() - epoch_start_time

            epoch_end_logs = {
                'avg_epoch_loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                'batches': num_batches,
                'global_batch': global_batch
            }
            self.trainer_state.update(epoch_end_logs)
            self.hooks.call_hooks('on_epoch_end', self.trainer_state)

        # Final metrics
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time
        training_metrics['total_global_batches'] = global_batch


        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self.hooks.call_hooks('on_train_end', self.trainer_state)

        return training_metrics

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate with accelerator."""
        logger.info("Starting evaluation...")
        if eval_dataloader is None:
            eval_dataloader = self.dataloader

        # Prepare eval dataloader if not already prepared
        if not hasattr(eval_dataloader, '_accelerator_prepared'):
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Evaluating", 
                                                       disable=not self.accelerator.is_local_main_process)):
                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Evaluation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    continue

                # Gather losses across all processes (simplified)
                batch_size = batch_data['input_ids'].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size * self.accelerator.num_processes
                num_batches_processed += 1

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        eval_metrics = {'val_loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['val_perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['val_perplexity'] = float('nan')

        self.model.train()

        logger.info(f"Evaluation results: Loss: {eval_metrics['val_loss']:.6f}, Perplexity: {eval_metrics['val_perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        
        return eval_metrics