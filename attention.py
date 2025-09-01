import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttentionSample:
    """Single attention analysis sample"""
    id: str
    sentences: List[str]
    question: str
    answer: str
    
    @classmethod
    def from_content(cls, sample_dict: Dict):
        content = sample_dict['content']
        # Handle escaped newlines in the JSON data
        content = content.replace('\\n', '\n')
        lines = content.strip().split('\n')
        
        sentences = []
        question = ""
        answer = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Sentence'):
                sentences.append(line.split(': ', 1)[1])
            elif line.startswith('Question'):
                question = line.split(': ', 1)[1]
            elif line.startswith('Answer'):
                answer = line.split(': ', 1)[1]
        
        return cls(
            id=sample_dict['id'],
            sentences=sentences,
            question=question,
            answer=answer
        )

def load_model_from_checkpoint(checkpoint_path, device, model_type):
    """Load model from checkpoint (adapted from run_inference_with_hooks)."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Try loading custom checkpoint format first
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # extract config
        if 'config' in checkpoint:
            config_data = checkpoint['config']
            # handle both dict and TransformerConfig object
            from src.config import TransformerConfig
            if hasattr(config_data, '__dict__'):
                config = config_data 
            else:
                config = TransformerConfig(**config_data)
        else:
            raise ValueError("No config found in checkpoint")
        
        logger.info(f"Model type: {model_type}")
        from src.model import get_model
        model = get_model(model_type, config)
        
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                # Try loading with strict=False to handle parameter mismatches
                logger.warning(f"Strict loading failed: {e}")
                logger.info("Attempting to load with strict=False...")
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # try loading directly if it's just the state dict
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError as e:
                # Try loading with strict=False to handle parameter mismatches
                logger.warning(f"Strict loading failed: {e}")
                logger.info("Attempting to load with strict=False...")
                model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Custom model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
        return model, config, 'custom'
        
    except Exception as e:
        logger.warning(f"Failed to load custom checkpoint: {e}")
        logger.info("Trying HuggingFace format...")
        
        # Fallback to HuggingFace format
        try:
            from transformers import AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained(checkpoint_path)
            model.to(device)
            model.eval()
            logger.info("HuggingFace model loaded successfully")
            return model, None, 'huggingface'
        except Exception as e2:
            raise ValueError(f"Could not load checkpoint in either custom or HuggingFace format. Custom error: {e}, HF error: {e2}")

def load_tokenizer(tokenizer_path_or_type):
    """Load tokenizer - handles both custom and HuggingFace formats."""
    try:
        if os.path.exists(tokenizer_path_or_type):
            # Try custom tokenizer first
            from src.mytokenizers import from_pretrained
            tokenizer = from_pretrained(tokenizer_path_or_type)
            logger.info(f"Loaded custom tokenizer from {tokenizer_path_or_type}")
            return tokenizer, 'custom'
        else:
            # Try HuggingFace tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_type)
            logger.info(f"Loaded HuggingFace tokenizer: {tokenizer_path_or_type}")
            return tokenizer, 'huggingface'
    except Exception as e1:
        try:
            # Try creating custom tokenizer
            from src.mytokenizers import create_tokenizer, add_reasoning_tokens
            tokenizer = create_tokenizer(tokenizer_path_or_type)
            tokenizer = add_reasoning_tokens(tokenizer)
            logger.info(f"Created custom tokenizer: {tokenizer_path_or_type}")
            return tokenizer, 'custom'
        except Exception as e2:
            raise ValueError(f"Could not load tokenizer. Path error: {e1}, Create error: {e2}")