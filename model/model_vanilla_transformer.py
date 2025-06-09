# ./model/model_vanilla_transformer.py
"""
Vanilla Transformer implementation for baseline comparison.
Standard transformer architecture following the original "Attention Is All You Need" design
with modern improvements like pre-layer normalization.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torch.serialization
from config import SymbolicConfig


class VanillaLayerNorm(nn.Module):
    """Standard layer normalization."""
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class VanillaCausalSelfAttention(nn.Module):
    """Standard causal self-attention mechanism."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Causal mask (lower triangular)
        self.register_buffer("causal_mask", 
                           torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores with scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        
        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        return y


class VanillaFeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class VanillaTransformerBlock(nn.Module):
    """Standard transformer block with pre-layer normalization."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = VanillaLayerNorm(config.n_embd, bias=config.bias)
        self.attn = VanillaCausalSelfAttention(config)
        self.ln_2 = VanillaLayerNorm(config.n_embd, bias=config.bias)
        self.ffn = VanillaFeedForward(config)

    def forward(self, x):
        # Pre-layer norm attention
        x = x + self.attn(self.ln_1(x))
        # Pre-layer norm feed-forward
        x = x + self.ffn(self.ln_2(x))
        return x


class VanillaTransformerModel(nn.Module):
    """
    Vanilla Transformer model for baseline comparison.
    Implements standard transformer architecture with:
    - Token and positional embeddings
    - Multi-head causal self-attention
    - Feed-forward networks
    - Pre-layer normalization
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)
        
        # Core model components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([VanillaTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=VanillaLayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to output projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"VanillaTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Architecture: vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wte'):
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the VanillaTransformerModel.
        
        Args:
            input_ids: Token input IDs (B, T)
            attention_mask: Attention mask (currently unused)
            labels: Target labels for language modeling loss (B, T)
            
        Returns:
            Dictionary containing loss and logits
        """
        device = input_ids.device
        b, t = input_ids.size()

        # Check sequence length limits
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")

        # Token embeddings
        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        
        # Positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        
        # Combine embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm and language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate language modeling loss if labels provided
        loss = None
        if labels is not None:
            # Validate labels
            if labels.shape != input_ids.shape:
                raise ValueError(f"Labels shape {labels.shape} must match input_ids shape {input_ids.shape}")
            
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy, >0 for stochastic)
            top_k: Top-k sampling parameter (None for full vocabulary)
            
        Returns:
            Generated token sequence (B, T + max_new_tokens)
        """
        # Input validation
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive if specified")
        
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Get logits for the last position and apply temperature
            logits = logits[:, -1, :] / max(temperature, 1e-7)  # Avoid division by zero
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            if temperature > 0:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Load a VanillaTransformerModel from a checkpoint.

        Args:
            checkpoint_path (str): Path to the .pt checkpoint file
            device (str): Device to load the model onto ('cpu', 'cuda', or specific device)

        Returns:
            tuple: (VanillaTransformerModel, tokenizer) or (None, None) if loading fails
        """
        # Validate inputs
        if not isinstance(checkpoint_path, str):
            print(f"Error: checkpoint_path must be a string, got {type(checkpoint_path)}")
            return None, None
        
        if not isinstance(device, str):
            print(f"Error: device must be a string, got {type(device)}")
            return None, None
        
        # Validate device
        try:
            target_device = torch.device(device)
        except Exception as e:
            print(f"Error: Invalid device '{device}': {e}")
            return None, None
        
        print(f"Loading checkpoint using {cls.__name__}.load_from_checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return None, None

        try:
            # Register safe globals for loading
            torch.serialization.add_safe_globals([SymbolicConfig])
            
            # Load checkpoint with proper error handling
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

            # Extract required components
            config = checkpoint.get('config')
            model_state_dict = checkpoint.get('model_state_dict')
            tokenizer = checkpoint.get('tokenizer')

            # Validate required components
            if not config:
                raise ValueError("Checkpoint missing required 'config' field")
            if not model_state_dict:
                raise ValueError("Checkpoint missing required 'model_state_dict' field")

            # Validate config type
            if not isinstance(config, SymbolicConfig):
                raise TypeError(f"Config must be SymbolicConfig, got {type(config)}")

            # Create and load model
            model = cls(config)
            
            # Load state dict with error handling
            try:
                model.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Loading state dict with strict=False due to: {e}")
                model.load_state_dict(model_state_dict, strict=False)
            
            # Move to target device and set evaluation mode
            model.to(target_device)
            model.eval()

            # Success message with model statistics
            param_count = model.get_num_params() / 1e6
            print(f"Model loaded successfully ({param_count:.2f}M parameters)")
            print(f"Model device: {next(model.parameters()).device}")
            
            if tokenizer:
                print(f"Tokenizer loaded successfully (type: {type(tokenizer).__name__})")
            else:
                print("Warning: Tokenizer not found in checkpoint")

            return model, tokenizer

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return None, None
        except PermissionError:
            print(f"Error: Permission denied accessing: {checkpoint_path}")
            return None, None
        except torch.serialization.pickle.UnpicklingError as e:
            print(f"Error: Failed to unpickle checkpoint: {e}")
            return None, None
        except RuntimeError as e:
            print(f"Error: PyTorch runtime error: {e}")
            return None, None
        except Exception as e:
            print(f"Error: Unexpected error during loading: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_checkpoint(self, checkpoint_path: str, tokenizer=None, additional_data=None):
        """
        Save model checkpoint with comprehensive error handling.
        
        Args:
            checkpoint_path (str): Path where to save the checkpoint
            tokenizer: Optional tokenizer to include in checkpoint
            additional_data (dict): Optional additional data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Input validation
        if not isinstance(checkpoint_path, str):
            print(f"Error: checkpoint_path must be a string, got {type(checkpoint_path)}")
            return False
        
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
            except Exception as e:
                print(f"Error: Failed to create directory {checkpoint_dir}: {e}")
                return False
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'config': self.config,
                'model_state_dict': self.state_dict(),
                'model_class': self.__class__.__name__,
                'pytorch_version': torch.__version__,
                'parameter_count': self.get_num_params(),
            }
            
            # Add optional components
            if tokenizer is not None:
                checkpoint_data['tokenizer'] = tokenizer
            
            if additional_data is not None:
                if isinstance(additional_data, dict):
                    checkpoint_data.update(additional_data)
                else:
                    print(f"Warning: additional_data must be dict, got {type(additional_data)}")
            
            # Atomic save using temporary file
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint_data, temp_path)
            
            # Move to final location
            os.replace(temp_path, checkpoint_path)
            
            print(f"Checkpoint saved successfully to: {checkpoint_path}")
            print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
            return True
            
        except Exception as e:
            print(f"Error: Failed to save checkpoint: {type(e).__name__}: {e}")
            
            # Clean up temporary file if it exists
            temp_path = checkpoint_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return False

    def get_model_info(self):
        """
        Return comprehensive model information for debugging and analysis.
        
        Returns:
            dict: Model information including parameters, configuration, and statistics
        """
        info = {
            'model_class': self.__class__.__name__,
            'total_parameters': self.get_num_params(),
            'non_embedding_parameters': self.get_num_params(non_embedding=True),
            'configuration': {
                'vocab_size': self.config.vocab_size,
                'n_embd': self.config.n_embd,
                'n_head': self.config.n_head,
                'n_layer': self.config.n_layer,
                'block_size': self.config.block_size,
                'dropout': self.config.dropout,
            },
            'architecture_type': 'Vanilla Transformer',
            'features': {
                'positional_encoding': 'Learned embeddings',
                'attention_type': 'Multi-head causal self-attention',
                'normalization': 'Pre-layer norm',
                'activation': 'GELU',
            },
            'architecture_details': {
                'head_dim': self.config.n_embd // self.config.n_head,
                'padding_idx': self.padding_idx,
                'ffn_hidden_size': 4 * self.config.n_embd,
            }
        }
        
        # Add device information if model has parameters
        if len(list(self.parameters())) > 0:
            info['device'] = str(next(self.parameters()).device)
            info['dtype'] = str(next(self.parameters()).dtype)
        
        return info


# Export the main model class
__all__ = ['VanillaTransformerModel']