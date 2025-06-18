# ./trainers/__init__.py
"""
Trainers module for cleanGPT.
"""

import torch
import logging
from typing import Dict, Type, Any, List, Optional

from .base_trainer import BaseTrainer
from .simple_trainer import SimpleTrainer
from .hooks import TrainingHook

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
                **kwargs) -> BaseTrainer:
    """Factory function to get trainer."""
    
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer type: '{trainer_type}'. Available: {available_trainers}")

    trainer_class = TRAINER_REGISTRY[trainer_type]
    
    return trainer_class(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
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
    'BaseTrainer', 'TrainingHook', 'SimpleTrainer', 
    'get_trainer', 'register_trainer', 'TRAINER_REGISTRY'
]

if ACCELERATE_AVAILABLE:
    __all__.append('AccelerateTrainer')