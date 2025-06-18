"""
Shared utilities for all transformer models.
"""

import torch
import torch.nn as nn
import math

class TransformerBase(nn.Module):
    """Shared foundation for all transformer variants."""
    
    # INITIALIZATION #

    def __init__(self, config):
        super().__init__()
        self.config = config
    
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

    #TODO implement
    @torch.no_grad()
    def generate(self):
        # generate new tokens autoregressively
        pass

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


    