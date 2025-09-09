#!/usr/bin/env python3
"""
Attention analysis for distractor datasets.
Tests how much attention target sentence tokens pay to relevant vs irrelevant context.

Usage:
python attention.py --checkpoint path/to/model.pt --data pair_data.json --output results.json
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

import torch
from tqdm import tqdm

# Import the model infrastructure
from src.model import get_model
from src.config import TransformerConfig
from src.mytokenizers import create_tokenizer, from_pretrained
from src.inference.hooks import InferenceHook, InferenceHookManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, device: str, model_type: str):
    """Load model from checkpoint - reusing logic from tests/run_inference_with_hooks.py"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    if 'config' in checkpoint:
        config_data = checkpoint['config']
        # Handle both dict and TransformerConfig object
        if hasattr(config_data, '__dict__'):
            config = config_data 
        else:
            config = TransformerConfig(**config_data)
    else:
        raise ValueError("No config found in checkpoint")
    
    model_type = model_type 
    logger.info(f"Model type: {model_type}")
    model = get_model(model_type, config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try loading directly if it's just the state dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")
    
    return model, config


def load_tokenizer(tokenizer_name: str):
    """Load tokenizer"""
    if os.path.exists(tokenizer_name):
        return from_pretrained(tokenizer_name)
    else:
        return create_tokenizer(tokenizer_name)


class AttentionTester:
    """Tests attention patterns for distractor analysis"""
    
    def __init__(self, data_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} distractor examples")
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load pair_data.json"""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def _concatenate_sentences(self, example: Dict) -> str:
        """Concatenate 4 context sentences + target sentence"""
        context_sentences = example['context_sentences']  # 4 sentences (2 relevant, 2 irrelevant)
        target_sentence = example['target_sentence']      # 1 target sentence
        
        # Concatenate all 5 sentences with periods and spaces
        all_sentences = context_sentences + [target_sentence]
        return ' '.join(sentence.strip() + ('.' if not sentence.strip().endswith('.') else '') for sentence in all_sentences)
    
    def _identify_relevant_positions(self, example: Dict, tokenized_sequence: List[str]) -> Tuple[List[int], List[int], List[int]]:
        """
        Identify which token positions correspond to:
        - Relevant context (from target story)
        - Irrelevant context (from distractor story) 
        - Target sentence
        """
        # Get the target story ID to identify relevant sentences
        target_story = example['target_story']  # e.g., "paired_000001_A"
        
        # Context sentences are mixed - need to figure out which are relevant
        context_sentences = example['context_sentences']
        target_sentence = example['target_sentence']
        
        # For this analysis, we need to know which context sentences came from target vs distractor
        # The fix_data.py script creates context by shuffling [T1, T2, C1, C2] where T=target, C=distractor
        # We'll need to tokenize each sentence separately to find positions
        
        relevant_positions = []
        irrelevant_positions = []
        target_positions = []
        
        current_pos = 0
        
        # Process each context sentence
        for sentence in context_sentences:
            sentence_tokens = self.tokenizer.encode(sentence.strip() + ('.' if not sentence.strip().endswith('.') else ''))
            sentence_length = len(sentence_tokens)
            
            # For now, we'll use a heuristic: check if sentence content suggests it's from target story
            # This is imperfect - ideally we'd track this in the data generation
            # TODO: Could improve by modifying fix_data.py to track source
            positions = list(range(current_pos, current_pos + sentence_length))
            
            # Simple heuristic: if this is one of the first 2 context sentences and target story ends with 'A',
            # assume some are relevant. This is a simplification.
            if len(relevant_positions) < 2 * sentence_length:  # Assume first half are relevant
                relevant_positions.extend(positions)
            else:
                irrelevant_positions.extend(positions)
            
            current_pos += sentence_length
        
        # Target sentence positions
        target_tokens = self.tokenizer.encode(target_sentence.strip() + ('.' if not target_sentence.strip().endswith('.') else ''))
        target_positions = list(range(current_pos, current_pos + len(target_tokens)))
        
        return relevant_positions, irrelevant_positions, target_positions
    
    def _extract_attention_weights(self, input_text: str) -> Dict:
        """Extract attention weights for the input sequence"""
        
        class AttentionCaptureHook(InferenceHook):
            """Captures all attention weights"""
            def __init__(self):
                super().__init__("attention_capture")
                self.attention_weights = {}
            
            def on_attention_computed(self, layer_idx, head_idx, attention_weights, 
                                    query, key, value, tokens, position, state):
                key = (layer_idx, head_idx)
                # Store the full attention matrix [seq_len, seq_len]
                self.attention_weights[key] = attention_weights[0].detach().cpu().numpy()
        
        # Setup hook
        hook = AttentionCaptureHook()
        hook_manager = InferenceHookManager()
        hook_manager.add_hook(hook)
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Run forward pass with hooks
        with torch.no_grad():
            # We need to modify the model forward to accept hook_manager
            # For now, let's call the model directly and handle attention extraction
            
            # This is a simplified approach - we'll extract attention from the model
            if hasattr(self.model, 'transformer'):
                # Standard transformer structure
                x = self.model.transformer.wte(input_tensor)  # Token embeddings
                if hasattr(self.model.transformer, 'wpe'):
                    pos = torch.arange(0, input_tensor.size(1), device=self.device).unsqueeze(0)
                    x = x + self.model.transformer.wpe(pos)
                
                # Process through layers
                for layer_idx, block in enumerate(self.model.transformer.h):
                    # Extract attention from this layer
                    if hasattr(block, 'attn'):
                        # Call attention with hooks
                        hook_state = {
                            'tokens': self.tokenizer.decode_tokens([input_ids]),
                            'position': input_tensor.size(1) - 1
                        }
                        x = block.attn(x, layer_idx=layer_idx, hook_manager=hook_manager, hook_state=hook_state)
                        
                        # Continue with rest of block
                        if hasattr(block, 'ln_1'):
                            x_norm = block.ln_1(x)
                        if hasattr(block, 'mlp'):
                            x = x + block.mlp(x_norm)
        
        return {
            'attention_weights': hook.attention_weights,
            'tokens': self.tokenizer.decode_tokens([input_ids]),
            'input_ids': input_ids
        }
    
    def test_model(self, model, tokenizer, model_name: str, max_examples: int = 100) -> Dict:
        """Test attention patterns on the distractor dataset"""
        self.model = model
        self.tokenizer = tokenizer
        
        results = {
            'model_name': model_name,
            'examples': [],
            'stats': {
                'total_samples': 0,
                'avg_relevant_attention': 0.0,
                'avg_irrelevant_attention': 0.0,
                'accuracy': 0.0  # Fraction where relevant > irrelevant
            }
        }
        
        relevant_scores = []
        irrelevant_scores = []
        correct_predictions = 0
        
        for i, example in enumerate(tqdm(self.data[:max_examples], desc="Testing attention")):
            try:
                # Concatenate sentences (4 context + 1 target)
                full_text = self._concatenate_sentences(example)
                
                # Extract attention weights
                attention_data = self._extract_attention_weights(full_text)
                tokens = attention_data['tokens']
                
                # Identify relevant/irrelevant/target positions
                # Note: This is simplified - ideally we'd have better source tracking
                relevant_pos, irrelevant_pos, target_pos = self._identify_relevant_positions(example, tokens)
                
                # Aggregate attention from target tokens to context
                total_relevant_attention = 0.0
                total_irrelevant_attention = 0.0
                total_target_tokens = 0
                
                # Sum attention across all layers and heads
                for (layer_idx, head_idx), attention_matrix in attention_data['attention_weights'].items():
                    seq_len = attention_matrix.shape[0]
                    
                    # For each target token, sum attention to relevant/irrelevant context
                    for target_token_pos in target_pos:
                        if target_token_pos < seq_len:
                            target_attention = attention_matrix[target_token_pos, :]
                            
                            # Sum attention to relevant positions
                            for pos in relevant_pos:
                                if pos < seq_len:
                                    total_relevant_attention += target_attention[pos]
                            
                            # Sum attention to irrelevant positions
                            for pos in irrelevant_pos:
                                if pos < seq_len:
                                    total_irrelevant_attention += target_attention[pos]
                            
                            total_target_tokens += 1
                
                # Average across target tokens
                if total_target_tokens > 0:
                    avg_relevant = total_relevant_attention / total_target_tokens
                    avg_irrelevant = total_irrelevant_attention / total_target_tokens
                    
                    relevant_scores.append(avg_relevant)
                    irrelevant_scores.append(avg_irrelevant)
                    
                    if avg_relevant > avg_irrelevant:
                        correct_predictions += 1
                    
                    # Store example result
                    results['examples'].append({
                        'id': example['id'],
                        'target_story': example['target_story'],
                        'relevant_attention': float(avg_relevant),
                        'irrelevant_attention': float(avg_irrelevant),
                        'prefers_relevant': avg_relevant > avg_irrelevant,
                        'text': full_text,
                        'target_positions': target_pos,
                        'relevant_positions': relevant_pos,
                        'irrelevant_positions': irrelevant_pos
                    })
                
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        # Compute final statistics
        if relevant_scores:
            results['stats']['total_samples'] = len(relevant_scores)
            results['stats']['avg_relevant_attention'] = float(np.mean(relevant_scores))
            results['stats']['avg_irrelevant_attention'] = float(np.mean(irrelevant_scores))
            results['stats']['accuracy'] = float(correct_predictions / len(relevant_scores))
            
            logger.info(f"Results for {model_name}:")
            logger.info(f"  Samples: {results['stats']['total_samples']}")
            logger.info(f"  Avg relevant attention: {results['stats']['avg_relevant_attention']:.4f}")
            logger.info(f"  Avg irrelevant attention: {results['stats']['avg_irrelevant_attention']:.4f}")
            logger.info(f"  Accuracy (relevant > irrelevant): {results['stats']['accuracy']:.2%}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test attention patterns on distractor dataset')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', default='pair_data.json', help='Path to distractor dataset')
    parser.add_argument('--output', default='attention_results.json', help='Output file for results')
    parser.add_argument('--model-type', default='vanilla', choices=['vanilla', 'tft', 'symbolic'])
    parser.add_argument('--tokenizer', default='gpt2', help='Tokenizer to use')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-examples', type=int, default=100, help='Maximum examples to test')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    device = args.device
    model, config = load_model_from_checkpoint(args.checkpoint, device, args.model_type)
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Create tester and run analysis
    tester = AttentionTester(args.data, device)
    results = tester.test_model(model, tokenizer, f"{args.model_type}_model", args.max_examples)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()