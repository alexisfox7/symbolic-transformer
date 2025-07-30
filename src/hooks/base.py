"""
Unified hook system for both training and inference.
Contains base hook, base training hook, HookManager
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from src.utils.logger import logger


class Hook(ABC):
    "Base class for all hooks"

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.data = {}
        logger.debug(f"Added hook: {self}")

    def __repr__(self):
        return f"Hook: {self.name}, Enabled: {self.enabled}"
    
    def clear(self):
        self.data = {}


class HookManager:
    "Manage hooks safely"

    def __init__(self):
        self.hooks: List[Hook] = []
    
    def add_hook(self, hook: Hook) -> None:
        self.remove_hook(hook.name)
        self.hooks.append(hook)
    
    def remove_hook(self, name: str) -> bool:
        hook = self.get_hook(name)
        if hook:
            self.hooks.remove(hook)
            return True
        return False

    def get_hook(self, name: str) -> Optional[Hook]:
        for hook in self.hooks:
            if hook.name == name:
                return hook
        return None
    
    def enable_hook(self, name: str) -> None:
        hook = self.get_hook(name)
        if hook:
            hook.enabled = True

    def disable_hook(self, name: str) -> None:
        hook = self.get_hook(name)
        if hook:
            hook.enabled = False

    def list_hooks(self):
        return [hook.name for hook in self.hooks]
    
    def call_hooks(self, method_name, *args, **kwargs):
        for hook in self.hooks:
            if not hook.enabled:
                continue
            
            method = getattr(hook, method_name, None)
            if method:
                method(*args, **kwargs)
    
class TrainingHook(Hook):
    def __init__(self, name):
        super().__init__(name)
    
    # core training events 
    def on_train_begin(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_train_end(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_begin(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        pass
    
    def on_batch_begin(self, state: Dict[str, Any]) -> None:
        pass

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        pass

class ModelHook(Hook):
    """Base class for model computation hooks (training or inference)."""
    
    def on_forward_begin(self, input_ids: torch.Tensor, state: Dict[str, Any]) -> None:
        """Called before model forward pass."""
        pass
    
    def on_forward_end(self, outputs: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Called after model forward pass."""
        pass
    
    def on_layer_begin(self, layer_idx: int, inputs: Any, state: Dict[str, Any]) -> None:
        """Called before processing a layer."""
        pass
    
    def on_layer_end(self, layer_idx: int, outputs: Any, state: Dict[str, Any]) -> None:
        """Called after processing a layer."""
        pass
    
    def on_attention_computed(self, layer_idx: int, attention_outputs: Dict[str, Any], state: Dict[str, Any]) -> None:
        """
        Called after attention computation.
        
        attention_outputs contains:
        - attention_weights: [batch, heads, seq_len, seq_len]
        - value_vectors: [batch, heads, seq_len, head_dim]
        - output: [batch, seq_len, hidden_dim]
        """
        pass
    
    def on_ffn_computed(self, layer_idx: int, ffn_outputs: Dict[str, Any], state: Dict[str, Any]) -> None:
        """
        Called after FFN computation.
        
        ffn_outputs contains:
        - input: [batch, seq_len, hidden_dim]
        - output: [batch, seq_len, hidden_dim]
        """
        pass

class InferenceHook(ModelHook):
    """Base class for inference-specific hooks."""
    
    def on_generation_begin(self, prompt: str, config: Dict[str, Any]) -> None:
        """Called at start of text generation."""
        pass
    
    def on_generation_step(self, step: int, token_id: int, state: Dict[str, Any]) -> None:
        """Called after each generation step."""
        pass
    
    def on_generation_end(self, generated_ids: List[int], state: Dict[str, Any]) -> None:
        """Called at end of generation."""
        pass
