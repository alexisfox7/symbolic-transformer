from typing import Any, Dict, List
import torch
from .base import InferenceHook


class AttentionExtractionHook(InferenceHook):
    """
    Extracts attention patterns for knowledge graph construction.
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
    
    def on_attention_computed(self, layer_idx: int, attention_outputs: Dict[str, Any], state: Dict[str, Any]) -> None:
        """
        Called after attention computation with new ModelHook interface.
        
        attention_outputs contains:
        - attention_weights: [batch, heads, seq_len, seq_len] 
        - query: [batch, heads, seq_len, head_dim]
        - key: [batch, heads, seq_len, head_dim]  
        - value: [batch, heads, seq_len, head_dim]
        - output: [batch, seq_len, hidden_dim]
        """
        attention_weights = attention_outputs.get('attention_weights')
        if attention_weights is None:
            return
            
        # Extract tokens and position from state
        tokens = state.get('tokens', [])
        position = state.get('position', 0)
        
        # Process each attention head
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        for head_idx in range(num_heads):
            head_weights = attention_weights[0, head_idx].detach().cpu()  # [seq_len, seq_len]
            
            # Ensure have enough tokens
            current_tokens = tokens.copy()
            while len(current_tokens) < seq_len:
                current_tokens.append(f"<POS_{len(current_tokens)}>")
            
            # Store edges above threshold for knowledge graph 
            significant_edges = []
            for i in range(seq_len):
                for j in range(seq_len):
                    weight = head_weights[i, j].item()
                    if weight > self.threshold:
                        edge_data = {
                            'source_pos': j,
                            'target_pos': i,  
                            'weight': weight,
                            'source_token': current_tokens[j], 
                            'target_token': current_tokens[i]   
                        }
                        significant_edges.append(edge_data)
            
            # Store record
            attention_record = {
                'layer': layer_idx,
                'head': head_idx,
                'position': position,
                'tokens': current_tokens[:seq_len],
                'edges': significant_edges,
                'attention_matrix': head_weights,
                'generation_context': {
                    'prompt_length': len(self.prompt_tokens) if hasattr(self, 'prompt_tokens') else 0,
                    'total_length': seq_len
                }
            }
            
            self.attention_data.append(attention_record)
            self.data.append(attention_record)

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