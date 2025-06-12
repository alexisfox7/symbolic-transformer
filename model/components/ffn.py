# Different kinds of FFN
import torch
import torch.nn as nn
from torch.nn import functional as F


class VanillaFFN(nn.Module):
    """Standard feed-forward network with GELU activation."""
    #TODO: old code
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    #TODO: old code
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class VocabFFN(nn.Module):
    """ Feed Forward Network (FFN) that constrains outputs to the vocabulary embedding manifold."""
    
    #TODO: implement
    def __init__(self, config):
        super().__init__()
        pass
    
    #TODO: implement
    def forward(self, x):
        pass

# add flag for use_v and use_proj
# add flag for kronecker (define function otuside since both ffn and attenion use it)
# add option for channel norm i/o, everywhere, vanilla
# add flag for residual

# notes
# make sure embedding sharing works
# fix config
