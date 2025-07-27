#src/trainers/base_trainer.py
"""
Updated base trainer with simple hook integration.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader

from .hooks import HookManager, TrainingHook

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer with hook system integration."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 output_dir: Optional[str] = None): 
        
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        
        # Hook system
        self.hooks = HookManager()

        self.hook_weights = {
            'early_exit': 0.5,
            'default': 0.5
        }
        
        # Training state (what hooks can access/modify)
        self.trainer_state = {
            'model': self.model,
            'optimizer': self.optimizer,
            'dataloader': self.dataloader,
            'device': self.device,
            'output_dir': self.output_dir,
            'model_params': self.model.get_num_params()
        }
            
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    # Hook management methods
    def add_hook(self, hook: TrainingHook) -> None:
        """Add a hook to the trainer."""
        self.hooks.add_hook(hook)
    
    def remove_hook(self, hook_name: str) -> bool:
        """Remove a hook by name."""
        return self.hooks.remove_hook(hook_name)
    
    def get_hook(self, hook_name: str) -> Optional[TrainingHook]:
        """Get a hook by name."""
        return self.hooks.get_hook(hook_name)
    
    # Convenience methods for adding common hooks
    def add_console_logging(self, log_every_n_batches: int = 10) -> None:
        """Add console logging hook."""
        from .hooks import ConsoleLogHook
        self.add_hook(ConsoleLogHook(log_every_n_batches))
    
    def add_json_logging(self, log_every_n_batches: int = 100) -> None:
        """Add JSON logging hook."""
        if not self.output_dir:
            logger.warning("No output_dir set - JSON logging disabled")
            return
        
        from .hooks import create_json_log_hook
        self.add_hook(create_json_log_hook(self.output_dir, log_every_n_batches))
    
    def add_checkpointing(self, save_every_n_epochs: int = 1) -> None:
        """Add checkpointing hook."""
        if not self.output_dir:
            logger.warning("No output_dir set - checkpointing disabled")
            return
        
        from .hooks import CheckpointHook
        self.add_hook(CheckpointHook(self.output_dir, save_every_n_epochs))
    
    def add_early_exiting(self) -> None:
        from .hooks import EarlyExitHook
        self.add_hook(EarlyExitHook())
        
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        Subclasses must implement this method.
        Should call hook methods at appropriate points.
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model."""
        pass
    
    def load_checkpoint(self, path: str, map_location: Optional[torch.device] = None):
        """Load a training checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        device = map_location or self.device
        checkpoint = torch.load(path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from: {path}")
        return checkpoint
    
    def collect_aux_losses(self):
        total_loss = 0

        for hook in self.hooks.hooks:
            if getattr(hook, 'get_aux_loss', None):
                loss, info = hook.get_aux_los()
                total_loss += self.hook_weights[info['loss_type']] * loss
        
        return total_loss
        