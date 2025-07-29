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
        self._parent_state = {}
        
    def set_hook_context(self, hook_manager, layer_idx: Optional[int] = None):
        """Set the hook manager and current layer index."""
        self._hook_manager = hook_manager
        self._current_layer_idx = layer_idx
        
    def set_parent_state(self, parent_state: Dict[str, Any]):
        """Set parent state containing input_ids and other context."""
        self._parent_state = parent_state
    
    def set_context(self, hook_manager, parent_state: Dict[str, Any], layer_idx: Optional[int] = None):
        """Combined method to set both hook context and parent state."""
        self.set_hook_context(hook_manager, layer_idx)
        self.set_parent_state(parent_state)
        
    def call_hooks(self, method_name: str, *args, **kwargs):
        """Call hooks if hook manager is available."""
        if self.has_hooks():
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