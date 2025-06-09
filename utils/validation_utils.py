# utils/validation_utils.py
"""
Simple validation utilities for train/val splits and validation during training.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_train_val_split(dataset, val_ratio: float = 0.1, seed: int = 42) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split dataset into train and validation sets.
    
    Args:
        dataset: Full dataset to split
        val_ratio: Fraction for validation (default: 0.1 = 10%)
        seed: Random seed for reproducible splits
        
    Returns:
        (train_dataset, val_dataset)
    """
    generator = torch.Generator().manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    return train_dataset, val_dataset


def run_validation(model: torch.nn.Module, 
                  val_dataloader: DataLoader, 
                  device: torch.device) -> Dict[str, float]:
    """
    Run validation and return metrics.
    
    Args:
        model: Model to evaluate
        val_dataloader: Validation data loader
        device: Device to run on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.get('loss')
            
            if loss is not None and not torch.isnan(loss):
                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
    model.train()  # Return to training mode
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    perplexity = torch.exp(torch.tensor(avg_loss)).item() if not torch.isnan(torch.tensor(avg_loss)) else float('nan')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'samples': total_samples
    }


# Modified data_utils.py function
def load_and_prepare_data_with_validation(dataset_name, dataset_config, tokenizer, max_samples, 
                                        max_seq_length, batch_size, val_ratio=0.1, 
                                        mlm=False, split='train', shuffle=True):
    """
    Load data and create train/validation split.
    
    Args:
        val_ratio: Fraction for validation (default: 0.1)
        ... (other args same as original)
        
    Returns:
        (train_dataloader, val_dataloader, tokenizer)
    """
    from utils.data_utils import load_and_prepare_data
    from torch.utils.data import DataLoader
    
    # Load full dataset
    full_dataloader, tokenizer = load_and_prepare_data(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        mlm=mlm,
        split=split,
        shuffle=False  # Don't shuffle yet, we'll split first
    )
    
    # Get the dataset from the dataloader
    full_dataset = full_dataloader.dataset
    
    # Create train/val split
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio)
    
    # Create separate dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=full_dataloader.collate_fn,
        drop_last=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        collate_fn=full_dataloader.collate_fn,
        drop_last=False,
        num_workers=0
    )
    
    logger.info(f"Created train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Created val dataloader: {len(val_dataloader)} batches")
    
    return train_dataloader, val_dataloader, tokenizer


# Modified simple_trainer.py training loop
class SimpleTrainerWithValidation:
    """
    Enhanced SimpleTrainer with validation support.
    """
    
    def __init__(self, *args, val_dataloader=None, validate_every=1, **kwargs):
        # Initialize base trainer (assuming it exists)
        super().__init__(*args, **kwargs)
        self.val_dataloader = val_dataloader
        self.validate_every = validate_every  # Validate every N epochs
        
    def train(self) -> Dict[str, Any]:
        """Training loop with validation."""
        from utils.validation_utils import run_validation
        
        logger.info("Starting training with validation...")
        
        # ... (existing training setup code)
        
        training_metrics = {
            'epoch_losses': [],
            'val_losses': [],      # Add validation losses
            'val_perplexities': [], # Add validation perplexities
            'final_loss': float('nan'),
            'final_val_loss': float('nan'),
            'training_time': 0.0,
        }
        
        for epoch in range(1, self.num_epochs + 1):
            # ... (existing training epoch code)
            
            # Training epoch
            epoch_loss = self._train_epoch(epoch)
            training_metrics['epoch_losses'].append(epoch_loss)
            
            # Validation epoch
            val_metrics = None
            if self.val_dataloader and (epoch % self.validate_every == 0):
                logger.info(f"Running validation for epoch {epoch}...")
                val_metrics = run_validation(self.model, self.val_dataloader, self.device)
                
                training_metrics['val_losses'].append(val_metrics['loss'])
                training_metrics['val_perplexities'].append(val_metrics['perplexity'])
                
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Enhanced epoch logging
            epoch_metrics = {'train_loss': epoch_loss}
            if val_metrics:
                epoch_metrics.update({
                    'val_loss': val_metrics['loss'],
                    'val_perplexity': val_metrics['perplexity']
                })
            
            self.log_epoch(epoch, epoch_loss, metrics=epoch_metrics)
            
            # Trigger callbacks with validation metrics
            self._trigger_callbacks('on_epoch_end', epoch, logs=epoch_metrics)
        
        # Final metrics
        if training_metrics['val_losses']:
            training_metrics['final_val_loss'] = training_metrics['val_losses'][-1]
        
        return training_metrics


# Enhanced JSON logging for validation
def log_validation_to_json(json_logger, epoch: int, val_metrics: Dict[str, float]):
    """Helper to log validation metrics to JSON."""
    if json_logger:
        json_logger.log_validation(epoch, val_metrics)


# Usage example in training script
def main_with_validation():
    """Example usage in training script."""
    # ... (setup code)
    
    # Load data with validation split
    train_dataloader, val_dataloader, tokenizer = load_and_prepare_data_with_validation(
        dataset_name="roneneldan/TinyStories",
        dataset_config=None,
        tokenizer=tokenizer,
        max_samples=10000,
        max_seq_length=128,
        batch_size=32,
        val_ratio=0.1  # 10% for validation
    )
    
    # Create trainer with validation
    trainer = SimpleTrainerWithValidation(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,  # Add validation dataloader
        optimizer=optimizer,
        device=device,
        validate_every=1,  # Validate every epoch
        num_epochs=5
    )
    
    # Train with validation
    results = trainer.train()
    
    logger.info(f"Final training loss: {results['final_loss']:.4f}")
    logger.info(f"Final validation loss: {results['final_val_loss']:.4f}")


# Simple plotting with validation curves
def plot_training_with_validation(training_metrics, save_path=None):
    """Plot training and validation curves."""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(training_metrics['epoch_losses']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_metrics['epoch_losses'], 'b-', label='Training Loss')
    if training_metrics['val_losses']:
        val_epochs = range(1, len(training_metrics['val_losses']) + 1)
        plt.plot(val_epochs, training_metrics['val_losses'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perplexity plot
    plt.subplot(1, 2, 2)
    if training_metrics['val_perplexities']:
        val_epochs = range(1, len(training_metrics['val_perplexities']) + 1)
        plt.plot(val_epochs, training_metrics['val_perplexities'], 'g-', label='Validation Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved to {save_path}")
    
    plt.show()