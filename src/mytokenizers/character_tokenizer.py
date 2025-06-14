# ./mytokenizers/character_tokenizer.py
"""
Character-level tokenizer implementation.
"""

import os
import json
from typing import List, Dict, Union, Optional, Any
import torch
from .base_tokenizer import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer that treats each character as a token."""
    
    def __init__(self, vocab=None, **kwargs):
        self.char_to_id = vocab or {}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()} if vocab else {}
        
        super().__init__(
            vocab_size=len(self.char_to_id),
            padding_token='<PAD>',
            eos_token='<EOS>',
            bos_token='<BOS>',
        )
        
        self._setup_special_tokens()
    
    def _setup_special_tokens(self):
        """Setup special tokens if not in vocab."""
        special_tokens = ['<PAD>', '<EOS>', '<BOS>', '<UNK>']
        
        for token in special_tokens:
            if token not in self.char_to_id:
                self.char_to_id[token] = len(self.char_to_id)
                self.id_to_char[len(self.id_to_char)] = token
        
        self.pad_token_id = self.char_to_id['<PAD>']
        self.eos_token_id = self.char_to_id['<EOS>']
        self.bos_token_id = self.char_to_id['<BOS>']
        self.unk_token_id = self.char_to_id['<UNK>']
        self.vocab_size = len(self.char_to_id)
    
    def build_vocab_from_texts(self, texts):
        """Build character vocabulary from text samples."""
        chars = set()
        for text in texts:
            chars.update(str(text))
        
        # Start with special tokens
        self.char_to_id = {'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<UNK>': 3}
        
        # Add characters
        for char in sorted(chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self._setup_special_tokens()
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into characters."""
        return list(str(text))
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, **kwargs) -> Union[List[int], List[List[int]]]:
        """Encode text to character IDs."""
        if isinstance(text, list):
            return [self.encode(t, add_special_tokens, **kwargs) for t in text]
        
        chars = self.tokenize(text)
        ids = [self.char_to_id.get(char, self.unk_token_id) for char in chars]
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False, **kwargs) -> str:
        """Decode character IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        chars = []
        for id in token_ids:
            char = self.id_to_char.get(id, '<UNK>')
            if skip_special_tokens and char in ['<PAD>', '<EOS>', '<BOS>', '<UNK>']:
                continue
            chars.append(char)
        
        return ''.join(chars)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs."""
        if isinstance(tokens, str):
            return self.char_to_id.get(tokens, self.unk_token_id)
        return [self.char_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to tokens."""
        if isinstance(ids, int):
            return self.id_to_char.get(ids, '<UNK>')
        return [self.id_to_char.get(id, '<UNK>') for id in ids]
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.char_to_id.copy()
