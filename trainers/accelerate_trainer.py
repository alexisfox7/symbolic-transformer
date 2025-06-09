# ./trainers/accelerate_trainer.py
"""
Simplified Accelerate Trainer without gradient accumulation.
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
    """Accelerate trainer without gradient accumulation complexity."""

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 callbacks: Optional[List[Callback]] = None):
        
        # Initialize accelerator (no gradient accumulation)
        self.accelerator = Accelerator(
            log_with=None,
            project_dir=output_dir
        )
        
        # Use accelerator's device instead of passed device
        super().__init__(model, dataloader, optimizer, self.accelerator.device, output_dir, callbacks)
        
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
        
        logger.info(f"AccelerateTrainer initialized:")
        logger.info(f"  Device: {self.accelerator.device}")
        logger.info(f"  Mixed precision: {self.accelerator.mixed_precision}")
        logger.info(f"  Log interval: every {log_interval} batches")
        if self.accelerator.num_processes > 1:
            logger.info(f"  Distributed training: {self.accelerator.num_processes} processes")

    def train(self) -> Dict[str, Any]:
        """Execute training loop with accelerate."""
        logger.info("Starting training with accelerate...")
        
        self.trainer_state['num_epochs'] = self.num_epochs
        self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

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

        for epoch in range(1, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
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
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

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
                if self.clip_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Track metrics
                batch_loss_item = loss.item()
                epoch_loss += batch_loss_item
                num_batches += 1
                global_batch += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                # Log at specified intervals
                if (batch_idx + 1) % self.log_interval == 0:
                    batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).shape[0]
                    samples_processed = (batch_idx + 1) * batch_size * self.accelerator.num_processes
                    self.log_batch(
                        batch_idx + 1, batch_loss_item, epoch=epoch,
                        metrics={
                            'samples': samples_processed,
                            'global_batch': global_batch,
                            'num_processes': self.accelerator.num_processes
                        }
                    )

                # Trigger batch end callback with global batch info
                self._trigger_callbacks('on_batch_end', batch_idx, logs={
                    'loss': batch_loss_item, 
                    'global_batch': global_batch
                })

            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['total_batches'] += num_batches
            training_metrics['total_samples'] += len(self.dataloader.dataset)

            epoch_duration = time.time() - epoch_start_time
            self.log_epoch(epoch, avg_epoch_loss, metrics={
                'duration': f"{epoch_duration:.2f}s",
                'batches': num_batches,
                'global_batch': global_batch
            })

            epoch_end_logs = {
                'loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                'batches': num_batches,
                'global_batch': global_batch
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
                    global_batch=global_batch
                )

        # Final metrics
        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time
        training_metrics['total_global_batches'] = global_batch

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Total batches processed: {global_batch}")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, **kwargs):
        """Save checkpoint using accelerator with timeout protection."""
        if not self.accelerator.is_main_process:
            return
            
        # Add timeout to prevent hanging
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Checkpoint save timed out")
        
        try:
            # Set 30 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            # Minimal wait - don't hang forever
            try:
                # Use a shorter timeout for wait_for_everyone
                self.accelerator.wait_for_everyone()
            except:
                logger.warning("Accelerator sync timed out, proceeding anyway")
            
            checkpoint = {
                'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                **kwargs
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save to temp file first
            temp_path = path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            # Atomic move
            os.replace(temp_path, path)
            logger.info(f"Checkpoint saved to: {path}")
            
        except TimeoutError:
            logger.error("Checkpoint save timed out!")
            raise
        finally:
            signal.alarm(0)  # Cancel timeout

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