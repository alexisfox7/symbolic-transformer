#./trainers/simple_trainer.py
"""
Simplified Trainer Implementation without Gradient Accumulation
Basic training loop with progress tracking and callback integration.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

#FIX:remove callback import, keep only basetrainer
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation without gradient accumulation.
    Provides straightforward training with progress tracking and callbacks.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10):
                 #FIX:remove callbacks parameter entirely
        """
        Initialize the simple trainer.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs (e.g., checkpoints).
            clip_grad_norm: Maximum norm for gradient clipping (None = no clipping).
            log_interval: Number of batches between logging.
        """
        #FIX:remove callbacks from parent call
        super().__init__(model, dataloader, optimizer, device, output_dir)
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        logger.info(f"SimpleTrainer initialized with {self.num_epochs} epochs.")
        logger.info(f"Batch size: {dataloader.batch_size}")
        if self.clip_grad_norm:
            logger.info(f"Gradient clipping enabled with max norm: {self.clip_grad_norm}")
        #FIX:remove callback logging, replace with hook logging
        if len(self.hooks.hooks) > 0:
            logger.info(f"Active hooks: {[h.name for h in self.hooks.hooks]}")

    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            A dictionary containing training metrics.
        """
        self.model.to(self.device)
        self.model.train()

        self.trainer_state['num_epochs'] = self.num_epochs
        self.trainer_state['is_main_process'] = True  # Simple trainer always runs on main process
        #FIX:replace _trigger_callbacks with hooks.on_train_begin
        self.hooks.on_train_begin(self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'),
            'training_time': 0.0,
            'total_batches': 0,
            'total_samples': 0
        }

        for epoch in range(1, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            #FIX:replace _trigger_callbacks with hooks.on_epoch_begin
            self.hooks.on_epoch_begin(epoch, self.trainer_state)
            epoch_start_time = time.time()
            
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                #FIX:remove _trigger_callbacks call for batch_begin - not needed

                #move batch to device
                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}

                #forward pass
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                    #FIX:remove _trigger_callbacks call
                    continue
                    
                if torch.isnan(loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                    self.trainer_state['status'] = 'NaN Loss'
                    #FIX:replace _trigger_callbacks with hooks.on_train_end
                    self.hooks.on_train_end(self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                #backward pass
                loss.backward()

                #gradient clipping if specified
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_grad_norm
                    )

                #optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                #track metrics
                batch_loss_item = loss.item()
                epoch_loss += batch_loss_item
                num_batches += 1

                #update progress bar
                progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                #log at specified intervals
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_size = batch.get('input_ids', next(iter(batch.values()))).shape[0]
                    samples_processed = (batch_idx + 1) * batch_size
                    #logging is handled by hooks now

                #FIX:replace _trigger_callbacks with hooks.on_batch_end and update state
                self.trainer_state['latest_loss'] = batch_loss_item
                self.hooks.on_batch_end(batch_idx, batch_loss_item, self.trainer_state)

            #calculate epoch metrics
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['total_batches'] += num_batches
            training_metrics['total_samples'] += len(self.dataloader.dataset)

            epoch_duration = time.time() - epoch_start_time
            
            epoch_end_logs = {
                'avg_loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                'batches': num_batches
            }
            self.trainer_state.update(epoch_end_logs)
            self.hooks.on_epoch_end(epoch, self.trainer_state)

        #final metrics
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Total batches processed: {training_metrics['total_batches']}")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        #FIX:replace _trigger_callbacks with hooks.on_train_end
        self.hooks.on_train_end(self.trainer_state)

        return training_metrics

    def log_batch(self,
                  batch_idx: int,
                  loss: float,
                  epoch: Optional[int] = None,
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training batch.

        Args:
            batch_idx (int): Index of the current batch.
            loss (float): Training loss for the batch.
            epoch (int, optional): Current epoch number.
            metrics (dict, optional): Additional metrics to log.
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())

        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        logger.info(f"{epoch_str}Batch {batch_idx}, Loss: {loss:.4f}" +
                   (f", {metrics_str}" if metrics_str else ""))

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.

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
        #FIX:replace _trigger_callbacks with hooks.on_evaluate_begin
        self.hooks.on_evaluate_begin(self.trainer_state)

        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                #FIX:remove _trigger_callbacks call for evaluate batch_begin - not needed

                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Evaluation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    #FIX:remove _trigger_callbacks call
                    continue

                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                num_batches_processed += 1

                #FIX:remove _trigger_callbacks call for evaluate batch_end - not needed

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        eval_metrics = {'loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['perplexity'] = float('nan')

        self.model.train()

        logger.info(f"Evaluation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        #FIX:replace _trigger_callbacks with hooks.on_evaluate_end
        self.hooks.on_evaluate_end(self.trainer_state)

        return eval_metrics