#different kinds of FFN
import torch
import torch.nn as nn
from torch.nn import functional as F
from sparsemax import Sparsemax

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
    """
    Vocabulary-constrained FFN that projects outputs back to vocabulary manifold.
    
    Implements modified formulation:
    1. z = FFN(x) = σ(xW^(1) + b^(1))W^(2) + b^(2)   [standard FFN]
    2. z_norm = LayerNorm(z)                         [normalize FFN output]
    3. similarities = z_norm @ E^T                   [compute similarities]
    4. probs = softmax(similarities)                 [full probability distribution]
    5. α = ReLU(probs - p) / sum(ReLU(probs - p))    [differentiable sparse thresholding]
    6. output = αE                                   [vocab projection]
    
    This ensures output ∈ conv(above-threshold vocabulary embeddings) with differentiability.
    """
    
    def __init__(self, config, vocab_embeddings_ref):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        
        self.vocab_embeddings_ref = vocab_embeddings_ref
        self.ffn = VanillaFFN(config)
        self.ffn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)

        # sparsity methods
        self.sparsity_method = "sparsemax" # "relu" or "sparsemax"
        self.prob_threshold = 0.1 # max 10 
        self.sparsemax = Sparsemax(dim=-1)
        
    def forward(self, x):
        """
        Forward pass with differentiable sparse thresholding.
        
        Args:
            x: Input tensor (B, T, n_embd)
            
        Returns:
            Vocabulary-grounded output tensor (B, T, n_embd)
        """
        if self.vocab_embeddings_ref is None:
            raise RuntimeError("vocab_embeddings_ref not set.")

        z = self.ffn(x)  # (B, T, n_embd)
        z_norm = self.ffn_norm(z)  # (B, T, n_embd)
        
        vocab_similarities = torch.matmul(z_norm, self.vocab_embeddings_ref.weight.T)  # (B, T, vocab_size)
        
        if self.sparsity_method == 'sparsemax':
            vocab_weights = self.sparsemax_method(vocab_similarities)
        elif self.sparsity_method == 'relu':
            vocab_weights = self.relu_method(vocab_similarities) # does softmax internally
        
        # project to vocabulary manifold (differentiable sparse combination)
        vocab_output = torch.matmul(vocab_weights, self.vocab_embeddings_ref.weight)  # (B, T, n_embd)
        
        return self.dropout(vocab_output)
    
    def relu_method(self, vocab_similarities):
        """
        ReLU threshold method with renormalization.
        """
        # full softmax
        full_probs = F.softmax(vocab_similarities, dim=-1)  # (B, T, vocab_size)
        
        # ReLU thresholding
        thresholded_probs = F.relu(full_probs - self.prob_threshold)  # (B, T, vocab_size)
        
        # renormalize to valid probability distribution
        prob_sums = thresholded_probs.sum(dim=-1, keepdim=True)  # (B, T, 1)
        
        # handle edge case where all probabilities are below threshold
        prob_sums = prob_sums + (prob_sums == 0).float() * 1e-8
        
        vocab_weights = thresholded_probs / prob_sums  # (B, T, vocab_size)
        
        return vocab_weights
    
    def sparsemax_method(self, vocab_similarities):
        return self.sparsemax(vocab_similarities)




# add flag for use_v and use_proj
# add flag for kronecker (define function otuside since both ffn and attenion use it)
# add option for channel norm i/o, everywhere, vanilla
# add flag for residual

# fix config
