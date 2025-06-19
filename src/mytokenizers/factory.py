#mytokenizers/factory.py
"""
Simple tokenizer factory
"""

import logging
from .gpt2_tokenizer import GPT2Tokenizer
from .character_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

TOKENIZERS = {
    'gpt2': GPT2Tokenizer,
    'character': CharacterTokenizer,
}

def create_tokenizer(tokenizer_type, **kwargs):
    """
    Create tokenizer instance.
    
    Args:
        tokenizer_type: 'gpt2' or 'character'
        **kwargs: Arguments passed to tokenizer constructor
        
    Returns:
        Tokenizer instance
    """
    if tokenizer_type not in TOKENIZERS:
        available = list(TOKENIZERS.keys())
        raise ValueError(f"Unknown tokenizer '{tokenizer_type}'. Available: {available}")
    
    tokenizer_class = TOKENIZERS[tokenizer_type]
    logger.info(f"Creating {tokenizer_type} tokenizer")
    return tokenizer_class(**kwargs)

def from_pretrained(tokenizer_type, directory_or_name, **kwargs):
    """
    Load pre-trained tokenizer.
    
    Args:
        tokenizer_type: 'gpt2' or 'character'  
        directory_or_name: Path or model name
        **kwargs: Additional arguments
        
    Returns:
        Tokenizer instance
    """
    if tokenizer_type not in TOKENIZERS:
        available = list(TOKENIZERS.keys())
        raise ValueError(f"Unknown tokenizer '{tokenizer_type}'. Available: {available}")
    
    tokenizer_class = TOKENIZERS[tokenizer_type]
    logger.info(f"Loading {tokenizer_type} tokenizer from {directory_or_name}")
    return tokenizer_class.from_pretrained(directory_or_name, **kwargs)