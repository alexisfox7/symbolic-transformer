# ./trainers/simple_trainer.py
"""
Simple Trainer Implementation with Gradient Accumulation Support
Basic training loop with progress tracking, callback integration, and gradient accumulation.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer, Callback

logger = logging.getLogger(__name__)

class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation with gradient accumulation support.
    This trainer provides a straightforward training process with
    progress tracking, basic logging, callback support, and efficient
    gradient accumulation for larger effective batch sizes.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 callbacks: Optional[List[Callback]] = None,
                 gradient_accumulation_steps: int = 1,
                 effective_batch_size: Optional[int] = None):
        """
        Initialize the simple trainer with gradient accumulation support.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs (e.g., checkpoints).
            clip_grad_norm: Maximum norm for gradient clipping (None = no clipping).
            log_interval: Number of gradient update steps between logging.
            callbacks: A list of Callback instances. Optional.
            gradient_accumulation_steps: Number of mini-batches to accumulate before updating.
            effective_batch_size: Target effective batch size. If provided, will calculate
                                 gradient_accumulation_steps automatically.
        """
        super().__init__(model, dataloader, optimizer, device, output_dir, callbacks)
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        # Calculate gradient accumulation steps
        if effective_batch_size is not None:
            mini_batch_size = dataloader.batch_size
            self.gradient_accumulation_steps = max(1, effective_batch_size // mini_batch_size)
            actual_effective_batch_size = self.gradient_accumulation_steps * mini_batch_size
            
            logger.info(f"Gradient accumulation configured:")
            logger.info(f"  Mini-batch size: {mini_batch_size}")
            logger.info(f"  Gradient accumulation steps: {self.gradient_accumulation_steps}")
            logger.info(f"  Effective batch size: {actual_effective_batch_size}")
            
            if actual_effective_batch_size != effective_batch_size:
                logger.warning(f"  Requested effective batch size ({effective_batch_size}) "
                             f"adjusted to {actual_effective_batch_size}")
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
            mini_batch_size = dataloader.batch_size
            actual_effective_batch_size = self.gradient_accumulation_steps * mini_batch_size
            
            logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            logger.info(f"Mini-batch size: {mini_batch_size}")
            logger.info(f"Effective batch size: {actual_effective_batch_size}")

        logger.info(f"SimpleTrainer initialized with {self.num_epochs} epochs.")
        if self.clip_grad_norm:
            logger.info(f"Gradient clipping enabled with max norm: {self.clip_grad_norm}")
        if self.callbacks:
            logger.info(f"Attached callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop with gradient accumulation.

        Returns:
            A dictionary containing training metrics.
        """
        logger.info("Starting training with gradient accumulation...")
        self.model.to(self.device)
        self.model.train()

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

        # Calculate total steps for progress tracking
        total_batches_per_epoch = len(self.dataloader)
        total_steps_per_epoch = (total_batches_per_epoch + self.gradient_accumulation_steps - 1) // self.gradient_accumulation_steps

        for epoch in range(1, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
            epoch_start_time = time.time()
            
            epoch_loss = 0.0
            accumulated_loss = 0.0
            num_accumulation_steps = 0
            num_updates = 0
            step_count = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue
                    
                if torch.isnan(loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                    self.trainer_state['status'] = 'NaN Loss'
                    self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                # Scale loss by accumulation steps to maintain proper averaging
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass (accumulate gradients)
                loss.backward()

                # Accumulate loss for logging
                batch_loss_item = loss.item() * self.gradient_accumulation_steps  # Unscale for logging
                accumulated_loss += batch_loss_item
                num_accumulation_steps += 1
                
                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})

                # Update parameters every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.dataloader):
                    # Clip gradients if specified
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.clip_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Calculate average loss for this accumulation window
                    avg_accumulated_loss = accumulated_loss / num_accumulation_steps
                    epoch_loss += avg_accumulated_loss
                    num_updates += 1
                    step_count += 1

                    # Update progress bar with current loss
                    progress_bar.set_postfix({
                        "loss": f"{avg_accumulated_loss:.4f}",
                        "step": f"{step_count}/{total_steps_per_epoch}"
                    })

                    # Log at specified intervals (based on update steps, not mini-batches)
                    if step_count % self.log_interval == 0:
                        samples_processed = (batch_idx + 1) * self.dataloader.batch_size
                        self.log_batch(
                            step_count, avg_accumulated_loss, epoch=epoch,
                            metrics={
                                'mini_batch': batch_idx + 1,
                                'samples': samples_processed,
                                'accumulation_steps': num_accumulation_steps
                            }
                        )

                    # Reset accumulation counters
                    accumulated_loss = 0.0
                    num_accumulation_steps = 0

            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / num_updates if num_updates > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['total_steps'] += step_count
            training_metrics['total_samples'] += len(self.dataloader.dataset)

            epoch_duration = time.time() - epoch_start_time
            self.log_epoch(epoch, avg_epoch_loss, metrics={
                'steps': step_count,
                'updates': num_updates,
                'duration': f"{epoch_duration:.2f}s"
            })

            epoch_end_logs = {
                'loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                'steps': step_count,
                'updates': num_updates
            }
            self.trainer_state.update(epoch_end_logs)
            self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

            # Save checkpoint
            if self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(
                    checkpoint_path, 
                    epoch=epoch, 
                    loss=avg_epoch_loss,
                    step=training_metrics['total_steps']
                )

        # Final metrics
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Total gradient updates: {training_metrics['total_steps']}")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def log_batch(self,
                  step_idx: int,
                  loss: float,
                  epoch: Optional[int] = None,
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training step (after gradient accumulation).

        Args:
            step_idx (int): Index of the current gradient update step.
            loss (float): Training loss for the accumulated mini-batches.
            epoch (int, optional): Current epoch number.
            metrics (dict, optional): Additional metrics to log.
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())

        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        logger.info(f"{epoch_str}Step {step_idx}, Loss: {loss:.4f}" +
                   (f", {metrics_str}" if metrics_str else ""))

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        Note: Evaluation doesn't use gradient accumulation.

        Args:
            eval_dataloader: DataLoader for evaluation data.

        Returns:
            A dictionary containing evaluation metrics.
        """
        logger.info("Starting evaluation...")
        if eval_dataloader is None:
            logger.warning("eval_dataloader not provided. Using training dataloader.")
            eval_dataloader = self.dataloader

        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
        self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Evaluation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue

                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size
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
