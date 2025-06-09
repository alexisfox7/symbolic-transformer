# ./trainers/__init__.py
"""
Trainers module for cleanGPT.

This module provides the necessary classes and factory functions for
creating and managing different training loop implementations (trainers)
and their associated callbacks, including gradient accumulation support.
"""

import torch
import logging
from typing import Dict, Type, Any, List, Optional

# Import base classes first
from .base_trainer import BaseTrainer, Callback

# Import concrete trainer implementations
from .simple_trainer import SimpleTrainer

logger = logging.getLogger(__name__)

# Registry of available trainer types
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'simple': SimpleTrainer,
}

def get_trainer(trainer_type: str,
                model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                callbacks: Optional[List[Callback]] = None,
                gradient_accumulation_steps: int = 1,
                effective_batch_size: Optional[int] = None,
                **kwargs) -> BaseTrainer:
    """
    Factory function to get an initialized trainer instance with gradient accumulation support.

    This function looks up the trainer_type in the TRAINER_REGISTRY
    and instantiates it with the provided arguments.

    Args:
        trainer_type (str): The type of trainer to use (e.g., 'simple').
                            Must be a key in TRAINER_REGISTRY.
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): The device (CPU or GPU) to train on.
        callbacks (Optional[List[Callback]]): A list of callback instances to
                                              be used during training. Defaults to None.
        gradient_accumulation_steps (int): Number of mini-batches to accumulate before
                                          updating parameters. Defaults to 1 (no accumulation).
        effective_batch_size (Optional[int]): Target effective batch size. If provided,
                                             will calculate gradient_accumulation_steps
                                             automatically based on dataloader.batch_size.
                                             Takes precedence over gradient_accumulation_steps.
        **kwargs: Additional keyword arguments specific to the chosen trainer type.
                  Common arguments might include 'num_epochs', 'output_dir', etc.

    Returns:
        An initialized instance of the requested trainer class.

    Raises:
        ValueError: If the trainer_type is not recognized (i.e., not in TRAINER_REGISTRY).
    """
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer type: '{trainer_type}'. "
            f"Available types: {available_trainers}"
        )

    trainer_class = TRAINER_REGISTRY[trainer_type]
    
    # Log gradient accumulation configuration
    if effective_batch_size is not None:
        mini_batch_size = dataloader.batch_size
        calculated_steps = max(1, effective_batch_size // mini_batch_size)
        logger.info(f"Initializing trainer '{trainer_type}' with:")
        logger.info(f"  Mini-batch size: {mini_batch_size}")
        logger.info(f"  Target effective batch size: {effective_batch_size}")
        logger.info(f"  Calculated accumulation steps: {calculated_steps}")
    elif gradient_accumulation_steps > 1:
        mini_batch_size = dataloader.batch_size
        effective_size = gradient_accumulation_steps * mini_batch_size
        logger.info(f"Initializing trainer '{trainer_type}' with:")
        logger.info(f"  Mini-batch size: {mini_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_size}")
    else:
        logger.info(f"Initializing trainer '{trainer_type}' without gradient accumulation")

    # Pass gradient accumulation parameters to trainer
    return trainer_class(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        callbacks=callbacks,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        **kwargs
    )

def register_trainer(name: str, trainer_class: Type[BaseTrainer]):
    """
    Register a new trainer type in the TRAINER_REGISTRY.

    This allows for extending the framework with custom trainer implementations
    that can then be accessed via get_trainer using the provided name.

    Args:
        name (str): The name (identifier) to register the trainer under.
        trainer_class (Type[BaseTrainer]): The trainer class to register.
                                           Must be a subclass of BaseTrainer.

    Raises:
        ValueError: If the name is already registered or if the
                    trainer_class is not a subclass of BaseTrainer.
    """
    if name in TRAINER_REGISTRY:
        raise ValueError(f"Trainer type '{name}' is already registered.")

    if not issubclass(trainer_class, BaseTrainer):
        raise ValueError(
            f"Trainer class '{trainer_class.__name__}' must inherit from BaseTrainer."
        )

    TRAINER_REGISTRY[name] = trainer_class
    logger.info(f"Registered new trainer type: '{name}' -> {trainer_class.__name__}")

# Define what is exported when 'from trainers import *' is used.
__all__ = [
    'BaseTrainer',
    'Callback',
    'SimpleTrainer',
    'get_trainer',
    'register_trainer',
    'TRAINER_REGISTRY'
]
