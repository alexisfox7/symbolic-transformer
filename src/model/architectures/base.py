"""
Shared utilities for all transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from src.inference.hooks import InferenceHookManager

class TransformerBase(nn.Module):
    """Shared foundation for all transformer variants."""
    
    # INITIALIZATION #

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hook_manager: Optional[InferenceHookManager] = None
    
    #REVIEW what are good default values for these?
    def _init_weights(self, module):
        """Standard transformer weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _apply_projection_init(self):
        """Apply special scaled initialization to output projections."""
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    #REVIEW - what are good default values for these?
    #NOTE currently not being used
    def configure_optimizer(self, weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95)):
        """Create optimizer with weight decay for appropriate parameters."""
        decay_params = set()
        no_decay_params = set()
        
        # common practice to not apply weight decay to biases and norm parameters
        #! make sure compatible with custom modules
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay_params.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay_params.add(fpn)
                else:
                    no_decay_params.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(decay_params)], 
                'weight_decay': weight_decay
            },
            {
                'params': [param_dict[pn] for pn in sorted(no_decay_params)], 
                'weight_decay': 0.0
            },
        ]
        
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
     
    # INFERENCE #

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, tokenizer=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting sequence of token ids [batch_size, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top k tokens
            tokenizer: Optional tokenizer for decoding tokens (needed for hooks)
            
        Returns:
            Generated token ids [batch_size, seq_len + max_new_tokens]
        """
        # Handle both 'input_ids' and 'idx' parameter names for compatibility
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        # Prepare generation state for hooks
        generation_state = {
            'temperature': temperature,
            'top_k': top_k,
            'max_new_tokens': max_new_tokens
        }
        
        # Get initial tokens for hooks
        tokens = []
        if tokenizer is not None and self.hook_manager is not None:
            tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
            self.hook_manager.on_generation_begin(tokens, generation_state)
            
        for position in range(max_new_tokens):
            # Crop sequence if it exceeds block size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            
            # Hook: before forward pass
            if self.hook_manager is not None:
                self.hook_manager.on_forward_begin(idx_cond, position, generation_state)
            
            # Get predictions
            outputs = self(idx_cond, hook_state={'position': position, 'tokens': tokens})
            logits = outputs['logits']
            
            # Hook: after forward pass
            if self.hook_manager is not None:
                self.hook_manager.on_forward_end(logits, position, generation_state)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            # Update tokens for hooks
            if tokenizer is not None and self.hook_manager is not None:
                tokens.append(tokenizer.decode([idx_next[0].item()]))
        
        # Hook: generation complete
        if self.hook_manager is not None and tokenizer is not None:
            final_tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
            self.hook_manager.on_generation_end(final_tokens, generation_state)
            
        return input_ids

    # UTILITY #

    #REVIEW check that it doesn't count the shared embeddings
    def get_num_params(self, non_embedding=True):
        """Count model parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    #TODO implement
    def get_model_info(self):
        # print comprehensive model info
        pass
    
    def set_hook_manager(self, hook_manager: Optional[InferenceHookManager]) -> None:
        """Set the inference hook manager for this model."""
        self.hook_manager = hook_manager


    