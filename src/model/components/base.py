"""
Base component class that provides hook management infrastructure.
"""
import torch.nn as nn
from typing import Dict, Any, Optional


class HookableComponent(nn.Module):
    """Base class for model components that support hooks."""
    
    def __init__(self):
        super().__init__()
        self._hook_manager = None
        self._current_layer_idx = None
        
    def set_hook_context(self, hook_manager, layer_idx: Optional[int] = None):
        """Set the hook manager and current layer index."""
        self._hook_manager = hook_manager
        self._current_layer_idx = layer_idx
        
    def call_hooks(self, method_name: str, *args, **kwargs):
        """Call hooks if hook manager is available."""
        if self._hook_manager is not None:
            self._hook_manager.call_hooks(method_name, *args, **kwargs)
            
    @property
    def hook_manager(self):
        """Access to hook manager."""
        return self._hook_manager
        
    @property
    def current_layer_idx(self):
        """Access to current layer index."""
        return self._current_layer_idx
        
    def has_hooks(self) -> bool:
        """Check if hook manager is available."""
        return self._hook_manager is not None