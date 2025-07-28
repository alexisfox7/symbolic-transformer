#./trainers/__init__.py
"""
Trainers module for cleanGPT.
"""

import torch
from typing import Dict, Type, Any, List, Optional
from accelerate.logging import get_logger

from .base_trainer import BaseTrainer
from .simple_trainer import SimpleTrainer
from .hooks import TrainingHook, HookManager
from .accelerate_trainer import AccelerateTrainer



logger = get_logger(__name__)

# registry of available trainer types
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'simple': SimpleTrainer,
    'accelerate': AccelerateTrainer
}

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

__all__ = [
    'BaseTrainer', 'TrainingHook', 'HookManager', 'SimpleTrainer', 'AccelerateTrainer',
    'get_trainer'
]
