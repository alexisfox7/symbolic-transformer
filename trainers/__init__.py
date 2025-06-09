# ./trainers/__init__.py
"""
Trainers module for cleanGPT.
"""

import torch
import logging
from typing import Dict, Type, Any, List, Optional

from .base_trainer import BaseTrainer, Callback
from .simple_trainer import SimpleTrainer

# Import accelerate trainer if available
try:
    from .accelerate_trainer import AccelerateTrainer
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    AccelerateTrainer = None

logger = logging.getLogger(__name__)

# Registry of available trainer types
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'simple': SimpleTrainer,
}

if ACCELERATE_AVAILABLE:
    TRAINER_REGISTRY['accelerate'] = AccelerateTrainer

def get_trainer(trainer_type: str,
                model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                callbacks: Optional[List[Callback]] = None,
                gradient_accumulation_steps: int = 1,
                effective_batch_size: Optional[int] = None,
                **kwargs) -> BaseTrainer:
    """Factory function to get trainer with accelerate support."""
    
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available: {available_trainers}")

    trainer_class = TRAINER_REGISTRY[trainer_type]
    
    # Handle effective batch size calculation
    if effective_batch_size is not None:
        mini_batch_size = dataloader.batch_size
        gradient_accumulation_steps = max(1, effective_batch_size // mini_batch_size)
        logger.info(f"Auto-calculated gradient accumulation steps: {gradient_accumulation_steps}")

    # For accelerate trainer, don't pass device (it handles this)
    if trainer_type == 'accelerate':
        return trainer_class(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,  # Still pass but accelerate will override
            callbacks=callbacks,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs
        )
    else:
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
    """Register new trainer type."""
    if name in TRAINER_REGISTRY:
        raise ValueError(f"Trainer type '{name}' is already registered.")
    if not issubclass(trainer_class, BaseTrainer):
        raise ValueError(f"Trainer class must inherit from BaseTrainer.")
    TRAINER_REGISTRY[name] = trainer_class
    logger.info(f"Registered trainer: '{name}'")

__all__ = [
    'BaseTrainer', 'Callback', 'SimpleTrainer', 
    'get_trainer', 'register_trainer', 'TRAINER_REGISTRY'
]

if ACCELERATE_AVAILABLE:
    __all__.append('AccelerateTrainer')