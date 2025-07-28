from random import randint
from typing import Any, Dict

import torch
import torch.nn as nn
from .base import ModelHook


class EarlyExitAnalysisHook(ModelHook):
    """Analyze model behavior for early exit training."""
    
    def __init__(self):
        super().__init__("early_exit_analysis")
        self.random_layer_idx = None
        self.exit_layer_output = None
        self.aux_loss = None
        self.model = None
        self.lm_head = None
        self.layer_norm = None
        self.targets = None
    
    def on_batch_begin(self, state: Dict[str, Any]) -> None:
        """Select random layer for this batch."""
        model = state.get('model')
        self.model = model
        
        if model:
            self.random_layer_idx = randint(0, model.config.n_layer - 1)
            self.exit_layer_output = None
            self.aux_loss = None
            self.lm_head = model.lm_head
            self.layer_norm = model.transformer.ln_f
            
            batch_data = state.get('current_batch', {})
            if 'targets' in batch_data:
                self.targets = batch_data['targets'].clone()
            elif 'input_ids' in batch_data:
                self.targets = batch_data['input_ids'].clone()
            else:
                raise ValueError("No targets or input_ids found in batch")
    
    def on_layer_end(self, layer_idx: int, outputs: Any, state: Dict[str, Any]) -> None:
        """Capture output from selected layer."""
        if layer_idx == self.random_layer_idx:
            self.exit_layer_output = outputs.clone()
            # Compute aux loss immediately after capture
            self._compute_aux_loss()
        
        # Set fallback zero loss if this is the last layer and nothing was captured
        if self.model and layer_idx == self.model.config.n_layer - 1:
            if self.aux_loss is None:
                device = outputs.device
                self.aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_aux_loss(self):
        """Compute auxiliary loss from captured layer output."""
        if self.exit_layer_output is None or self.targets is None:
            return
        
        # Apply layer norm and compute logits
        normalized_output = self.layer_norm(self.exit_layer_output)
        logits = self.lm_head(normalized_output)
        
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_targets = self.targets[:, 1:].contiguous().view(-1)
        
        self.aux_loss = loss_func(shift_logits, shift_targets)
    
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        """Ensure aux loss is set."""
        if self.aux_loss is None:
            device = state.get('device', torch.device('cpu'))
            self.aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    def get_aux_loss(self):
        """Return auxiliary loss + what type it is."""
        if self.aux_loss is None:
            raise ValueError("Auxiliary loss is empty")
        
        info = {'loss_type': 'early_exit'}
        return self.aux_loss, info
