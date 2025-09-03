# src/inference/hooks.py
"""
Inference hook system for extracting model internals during generation.
"""

from typing import Dict, Any, List, Optional
import torch
from accelerate.logging import get_logger

logger = get_logger(__name__)


class InferenceHook:
    """
    Base class for inference-time hooks that observe model computations during forward passes.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.data = [] 
    
    def __repr__(self):
        return f"InferenceHook('{self.name}', enabled={self.enabled})"
    
    def clear(self):
        """Clear accumulated data"""
        self.data = []
    
    # core inference events
    def on_generation_begin(self, prompt_tokens: List[str], state: Dict[str, Any]) -> None:
        """Called at start of generation"""
        pass
    
    def on_generation_end(self, generated_tokens: List[str], state: Dict[str, Any]) -> None:
        """Called at end of generation"""
        pass
    
    def on_forward_begin(self, input_ids: torch.Tensor, position: int, state: Dict[str, Any]) -> None:
        """Called before each forward pass during generation"""
        pass
    
    def on_forward_end(self, logits: torch.Tensor, position: int, state: Dict[str, Any]) -> None:
        """Called after each forward pass during generation"""
        pass
    
    def on_attention_computed(self, layer_idx: int, head_idx: int, 
                            attention_weights: torch.Tensor, 
                            query: torch.Tensor,
                            key: torch.Tensor, 
                            value: torch.Tensor,
                            tokens: List[str],
                            position: int,
                            state: Dict[str, Any]) -> None:
        """Called after attention computation in each head"""
        pass
    
    def on_ffn_computed(self, layer_idx: int, 
                       ffn_input: torch.Tensor,
                       ffn_output: torch.Tensor, 
                       tokens: List[str],
                       position: int,
                       state: Dict[str, Any]) -> None:
        """Called after FFN computation"""
        pass


class InferenceHookManager:
    """Manages inference hooks during model forward passes"""
    
    def __init__(self):
        self.hooks: List[InferenceHook] = []
        self.logger = logging.getLogger(__name__)
    
    def add_hook(self, hook: InferenceHook) -> None:
        """Add a hook, replacing any existing hook with same name"""
        self.hooks = [h for h in self.hooks if h.name != hook.name]
        self.hooks.append(hook)
        self.logger.info(f"Added inference hook: {hook.name}")
    
    def remove_hook(self, name: str) -> bool:
        """Remove hook by name"""
        original_len = len(self.hooks)
        self.hooks = [h for h in self.hooks if h.name != name]
        removed = len(self.hooks) < original_len
        if removed:
            self.logger.info(f"Removed inference hook: {name}")
        return removed
    
    def get_hook(self, name: str) -> Optional[InferenceHook]:
        """Get hook by name"""
        for hook in self.hooks:
            if hook.name == name:
                return hook
        return None
    
    def clear_all_data(self):
        """Clear data from all hooks"""
        for hook in self.hooks:
            hook.clear()
    
    def list_hooks(self) -> List[str]:
        """List all hook names"""
        return [h.name for h in self.hooks]
    
    def _call_hook_method(self, method_name: str, *args, **kwargs) -> None:
        """Safely call a method on all enabled hooks"""
        if not self.hooks:
            return
        
        for hook in self.hooks:
            if not hook.enabled:
                continue
            
            try:
                method = getattr(hook, method_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Inference hook {hook.name}.{method_name} failed: {e}")
    
    # convenience methods for models to call
    def on_generation_begin(self, prompt_tokens: List[str], state: Dict[str, Any]) -> None:
        self._call_hook_method('on_generation_begin', prompt_tokens, state)
    
    def on_generation_end(self, generated_tokens: List[str], state: Dict[str, Any]) -> None:
        self._call_hook_method('on_generation_end', generated_tokens, state)
    
    def on_forward_begin(self, input_ids: torch.Tensor, position: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_forward_begin', input_ids, position, state)
    
    def on_forward_end(self, logits: torch.Tensor, position: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_forward_end', logits, position, state)
    
    def on_attention_computed(self, layer_idx: int, head_idx: int, 
                            attention_weights: torch.Tensor, 
                            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            tokens: List[str], position: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_attention_computed', layer_idx, head_idx, 
                             attention_weights, query, key, value, tokens, position, state)
    
    def on_ffn_computed(self, layer_idx: int, ffn_input: torch.Tensor, ffn_output: torch.Tensor, 
                       tokens: List[str], position: int, state: Dict[str, Any]) -> None:
        self._call_hook_method('on_ffn_computed', layer_idx, ffn_input, ffn_output, tokens, position, state)




class FFNActivationTracker(InferenceHook):
    """Hook for tracking FFN activations and norms"""
    
    def __init__(self, layers_to_track: List[int] = None):
        super().__init__("ffn_activation_tracker")
        self.layers_to_track = layers_to_track or []
        self.activations = []
    
    def on_ffn_computed(self, layer_idx, ffn_input, ffn_output, tokens, position, state):
        if not self.layers_to_track or layer_idx in self.layers_to_track:
            self.activations.append({
                'layer': layer_idx,
                'position': position,
                'input_norm': ffn_input.norm().item(),
                'output_norm': ffn_output.norm().item(),
                'tokens': tokens.copy()
            })