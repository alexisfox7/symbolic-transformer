# src/inference/hooks.py
"""
Inference hook system for extracting model internals during generation.
Parallel to training hooks but focused on model activations and computations.
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import logging

logger = logging.getLogger(__name__)


class InferenceHook:
    """
    Base class for inference-time hooks that observe model internals.
    
    Unlike training hooks (which observe trainer state), inference hooks
    observe model computations during forward passes.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.data = []  # Most hooks will accumulate data
    
    def __repr__(self):
        return f"InferenceHook('{self.name}', enabled={self.enabled})"
    
    def clear(self):
        """Clear accumulated data"""
        self.data = []
    
    # Core inference events
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
    
    # Convenience methods for models to call
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


class AttentionExtractionHook(InferenceHook):
    """
    Extracts attention patterns for knowledge graph construction.
    
    Stores significant attention edges that can later be used to build
    attention-based knowledge graphs.
    """
    
    def __init__(self, threshold: float = 0.1, store_values: bool = False):
        super().__init__("attention_extractor")
        self.threshold = threshold
        self.store_values = store_values
        self.attention_data = []
    
    def clear(self):
        """Clear accumulated attention data"""
        self.attention_data = []
        self.data = []
    
    def on_generation_begin(self, prompt_tokens: List[str], state: Dict[str, Any]) -> None:
        """Store prompt context"""
        self.prompt_tokens = prompt_tokens
        self.generation_state = state.copy()
    
    def on_attention_computed(self, layer_idx: int, head_idx: int, 
                            attention_weights: torch.Tensor, 
                            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            tokens: List[str], position: int, state: Dict[str, Any]) -> None:
        """Extract significant attention edges"""
        
        # attention_weights shape: [batch_size, seq_len, seq_len]
        # We typically only care about batch_size=1 during generation
        att_matrix = attention_weights[0].detach().cpu()  # [seq_len, seq_len]
        
        # Find significant edges (above threshold)
        significant_edges = []
        seq_len = att_matrix.shape[0]
        
        for i in range(seq_len):
            for j in range(seq_len):
                weight = att_matrix[i, j].item()
                if weight > self.threshold:
                    edge_data = {
                        'source_pos': j,  # what we're attending FROM
                        'target_pos': i,  # what we're attending TO  
                        'weight': weight,
                        'source_token': tokens[j] if j < len(tokens) else '<UNK>',
                        'target_token': tokens[i] if i < len(tokens) else '<UNK>'
                    }
                    
                    # Optionally store value vectors for deeper analysis
                    if self.store_values:
                        edge_data['value_vector'] = value[0, head_idx, j, :].detach().cpu()
                    
                    significant_edges.append(edge_data)
        
        # Store this layer/head's attention data
        attention_record = {
            'layer': layer_idx,
            'head': head_idx,
            'position': position,  # which generation step
            'tokens': tokens.copy(),
            'edges': significant_edges,
            'attention_matrix': att_matrix,  # full matrix for analysis
            'generation_context': {
                'prompt_length': len(self.prompt_tokens),
                'total_length': len(tokens)
            }
        }
        
        self.attention_data.append(attention_record)
        self.data.append(attention_record)  # for base class compatibility
    
    def get_edges_for_layer_head(self, layer: int, head: int) -> List[Dict]:
        """Get all edges for a specific layer/head combination"""
        edges = []
        for record in self.attention_data:
            if record['layer'] == layer and record['head'] == head:
                edges.extend(record['edges'])
        return edges
    
    def get_token_attention_summary(self, token: str) -> Dict[str, Any]:
        """Get summary of how much attention a token receives/gives"""
        received_attention = []
        given_attention = []
        
        for record in self.attention_data:
            for edge in record['edges']:
                if edge['target_token'] == token:
                    received_attention.append({
                        'from': edge['source_token'],
                        'weight': edge['weight'],
                        'layer': record['layer'],
                        'head': record['head']
                    })
                if edge['source_token'] == token:
                    given_attention.append({
                        'to': edge['target_token'],
                        'weight': edge['weight'],
                        'layer': record['layer'],
                        'head': record['head']
                    })
        
        return {
            'token': token,
            'received_attention': received_attention,
            'given_attention': given_attention,
            'total_received': sum(a['weight'] for a in received_attention),
            'total_given': sum(a['weight'] for a in given_attention)
        }


# Factory functions
def create_attention_extraction_hook(threshold: float = 0.1, store_values: bool = False) -> AttentionExtractionHook:
    """Create an attention extraction hook"""
    return AttentionExtractionHook(threshold, store_values)


# Example of other hooks you might want:
class SymbolicStreamHook(InferenceHook):
    """Hook for tracking symbolic vs contextual stream activations"""
    
    def __init__(self):
        super().__init__("symbolic_stream_tracker")
        self.stream_data = []
    
    def on_attention_computed(self, layer_idx, head_idx, attention_weights, 
                            query, key, value, tokens, position, state):
        # For symbolic transformers, track which stream the attention operates on
        if 'stream_type' in state:  # Assuming models provide this info
            self.stream_data.append({
                'layer': layer_idx,
                'head': head_idx,
                'position': position,
                'stream_type': state['stream_type'],  # 'symbolic' or 'contextual'
                'attention_norm': attention_weights.norm().item()
            })


class ActivationHook(InferenceHook):
    """Hook for tracking intermediate activations"""
    
    def __init__(self, layers_to_track: List[int] = None):
        super().__init__("activation_tracker")
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