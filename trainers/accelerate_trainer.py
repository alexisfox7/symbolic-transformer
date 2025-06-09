# ./trainers/accelerate_trainer.py
"""
Fixed Accelerate Trainer with proper gradient accumulation handling.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

from .base_trainer import BaseTrainer, Callback

logger = logging.getLogger(__name__)

class AccelerateTrainer(BaseTrainer):
    """Accelerate trainer with proper gradient accumulation handling."""

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,  # This is now in terms of gradient update steps
                 callbacks: Optional[List[Callback]] = None,
                 gradient_accumulation_steps: int = 1):
        
        # Initialize accelerator first
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=None,
            project_dir=output_dir
        )
        
        # Use accelerator's device instead of passed device
        super().__init__(model, dataloader, optimizer, self.accelerator.device, output_dir, callbacks)
        
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval  # Now refers to gradient update steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
        
        logger.info(f"AccelerateTrainer initialized:")
        logger.info(f"  Device: {self.accelerator.device}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {self.accelerator.mixed_precision}")
        logger.info(f"  Log interval: every {log_interval} gradient update steps")
        if self.accelerator.num_processes > 1:
            logger.info(f"  Distributed training: {self.accelerator.num_processes} processes")

    def train(self) -> Dict[str, Any]:
        """Execute training loop with proper accelerate gradient accumulation."""
        logger.info("Starting training with accelerate...")
        
        self.trainer_state['num_epochs'] = self.num_epochs
        self.trainer_state['gradient_accumulation_steps'] = self.gradient_accumulation_steps
        self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'),
            'training_time': 0.0,
            'total_steps': 0,
            'total_samples': 0
        }

        # Track gradient update steps separately from mini-batches
        global_step = 0

        for epoch in range(1, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
            epoch_start_time = time.time()
            
            epoch_loss = 0.0
            num_gradient_updates = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
                disable=not self.accelerator.is_local_main_process
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Use accelerator's accumulation context - this handles when to actually step
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch_data)
                    loss = outputs.get('loss')

                    if loss is None:
                        logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                        continue
                        
                    if torch.isnan(loss):
                        logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                        self.trainer_state['status'] = 'NaN Loss'
                        self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                        training_metrics['training_time'] = time.time() - total_start_time
                        return training_metrics

                    # Backward pass - accelerator handles gradient scaling
                    self.accelerator.backward(loss)

                    # Gradient clipping if specified
                    if self.accelerator.sync_gradients and self.clip_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Optimizer step - accelerator.accumulate context handles whether to actually step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Check if a gradient update actually happened
                # This is tricky with accelerate, but we can track it by checking if we're at accumulation boundary
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    global_step += 1
                    num_gradient_updates += 1
                    
                    # Track metrics for actual gradient updates
                    batch_loss_item = loss.item()
                    epoch_loss += batch_loss_item

                    # Update progress bar with current loss and step info
                    progress_bar.set_postfix({
                        "loss": f"{batch_loss_item:.4f}",
                        "step": f"{global_step}",
                        "accum": f"{(batch_idx + 1) % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}"
                    })

                    # Log at gradient update intervals (not mini-batch intervals)
                    if global_step % self.log_interval == 0:
                        samples_processed = (batch_idx + 1) * self.dataloader.batch_size * self.accelerator.num_processes
                        self.log_batch(
                            global_step, batch_loss_item, epoch=epoch,
                            metrics={
                                'samples': samples_processed,
                                'mini_batch': batch_idx + 1,
                                'effective_batch_size': self.dataloader.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes
                            }
                        )

                # Always trigger batch end callback (for compatibility)
                batch_loss_item = loss.item() if loss is not None else None
                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item, 'global_step': global_step})

            # Handle final partial accumulation at end of epoch
            if len(self.dataloader) % self.gradient_accumulation_steps != 0:
                global_step += 1
                num_gradient_updates += 1
                epoch_loss += batch_loss_item if 'batch_loss_item' in locals() else 0.0

            # Calculate epoch metrics based on gradient updates
            avg_epoch_loss = epoch_loss / num_gradient_updates if num_gradient_updates > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['total_steps'] += num_gradient_updates
            training_metrics['total_samples'] += len(self.dataloader.dataset)

            epoch_duration = time.time() - epoch_start_time
            self.log_epoch(epoch, avg_epoch_loss, metrics={
                'duration': f"{epoch_duration:.2f}s",
                'gradient_updates': num_gradient_updates,
                'global_step': global_step
            })

            epoch_end_logs = {
                'loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                'gradient_updates': num_gradient_updates,
                'global_step': global_step
            }
            self.trainer_state.update(epoch_end_logs)
            self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

            # Save checkpoint (only on main process)
            if self.output_dir and self.accelerator.is_main_process:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(
                    checkpoint_path, 
                    epoch=epoch, 
                    loss=avg_epoch_loss,
                    step=global_step
                )

        # Final metrics
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time
        training_metrics['total_gradient_steps'] = global_step

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Total gradient updates: {global_step}")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, **kwargs):
        """Save checkpoint using accelerator."""
        if not self.accelerator.is_main_process:
            return
            
        self.accelerator.wait_for_everyone()
        
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'accelerator_state': self.accelerator.get_state_dict(),
            **kwargs
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate with accelerator."""
        logger.info("Starting evaluation...")
        if eval_dataloader is None:
            eval_dataloader = self.dataloader

        # Prepare eval dataloader if not already prepared
        if not hasattr(eval_dataloader, '_accelerator_prepared'):
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
        self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Evaluating", 
                                                       disable=not self.accelerator.is_local_main_process)):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Evaluation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    continue

                # Gather losses across all processes
                all_losses = self.accelerator.gather(loss.repeat(batch_data['input_ids'].shape[0]))
                
                batch_size = all_losses.shape[0]
                total_loss += all_losses.sum().item()
                total_samples += batch_size
                num_batches_processed += 1

                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': loss.item()})

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        eval_metrics = {'loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['perplexity'] = float('nan')

        self.model.train()

        logger.info(f"Evaluation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        self._trigger_callbacks('on_evaluate_end', logs=self.trainer_state)

        return eval_metrics