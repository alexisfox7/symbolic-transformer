#!/usr/bin/env python
# ./examples/train_symbolic_example.py
"""
Example script demonstrating how to train a Pure Symbolic Transformer model with ALiBi.
This script shows training a model where all internal representations are constrained
to remain symbolically interpretable as combinations of vocabulary embeddings.

Key features demonstrated:
- Pure symbolic reasoning through vocabulary-constrained operations
- Channel-wise layer normalization preserving head structure
- Vocabulary-grounded FFN projections ensuring interpretability
- ALiBi positional encoding maintaining symbolic purity

Usage examples:
python ./examples/train_symbolic_example.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 100000 \
  --preset small \
  --block_size 128 \
  --batch_size 64 \
  --num_epochs 5 \
  --output_dir "./outputs/symbolic_test" \
  --test_generation

python ./examples/train_symbolic_example.py \
  --dataset "wikimedia/wikipedia" \
  --dataset_config "20231101.en" \
  --preset medium \
  --max_samples 50000 \
  --batch_size 32 \
  --num_epochs 8 \
  --use_symbolic_ffn \
  --use_vocab_refinement \
  --output_dir "./outputs/symbolic_wiki"
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime
import json

# Add the parent directory to sys.path to access the cleanGPT modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset


class TeeOutput:
    """Class to redirect output to both console and file simultaneously."""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8', buffering=1)
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()


def setup_logging(output_dir):
    """Set up comprehensive logging to both console and files."""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log_path = os.path.join(logs_dir, f'symbolic_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(training_log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    output_log_path = os.path.join(logs_dir, f'symbolic_output_{timestamp}.log')
    tee_output = TeeOutput(output_log_path)
    
    return logger, tee_output, logs_dir, timestamp


def test_symbolic_interpretability(model, tokenizer, device, config, output_dir, timestamp):
    """
    Test the symbolic interpretability capabilities of the model.
    This demonstrates the key benefit of pure symbolic constraints.
    """
    print("\n" + "="*60)
    print("TESTING SYMBOLIC INTERPRETABILITY")
    print("="*60)
    
    # Create interpretability test log file
    interp_log_path = os.path.join(output_dir, 'logs', f'symbolic_interpretability_{timestamp}.log')
    
    model.eval()
    
    # Test prompts that require symbolic reasoning
    test_prompts = [
        # Story completion requiring logical inference
        "Max went to",
        "The key opens the door. Sarah has the key. Sarah can",
        "Rain makes things wet. It is raining. The ground is",
    ]
    
    with open(interp_log_path, 'w', encoding='utf-8') as interp_file:
        interp_file.write(f"Symbolic Interpretability Test Results\n")
        interp_file.write(f"Timestamp: {timestamp}\n")
        interp_file.write(f"Model: Symbolic Transformer\n")
        interp_file.write(f"Vocab size: {config.vocab_size}\n")
        interp_file.write(f"All internal representations constrained to vocabulary manifold\n")
        interp_file.write("="*80 + "\n\n")
        
        for i, prompt in enumerate(test_prompts):
            test_result = f"\n--- Symbolic Test {i+1} ---\n"
            test_result += f"Prompt: \"{prompt}\"\n"
            
            print(test_result.strip())
            interp_file.write(test_result)
            
            # Tokenize prompt
            input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], device=device)
            prompt_length = input_ids.size(1)
            
            length_info = f"Prompt tokens: {prompt_length}\n"
            print(length_info.strip())
            interp_file.write(length_info)
            
            # Test different generation lengths
            for gen_len in [5, 10, 20]:
                gen_info = f"\nGenerating {gen_len} tokens:\n"
                print(gen_info.strip())
                interp_file.write(gen_info)
                
                try:
                    with torch.no_grad():
                        # Generate with low temperature for more deterministic symbolic reasoning
                        generated = model.generate(
                            input_ids, 
                            max_new_tokens=gen_len,
                            temperature=0.3,  # Low temperature for symbolic reasoning
                            top_k=20
                        )
                    
                    # Decode the generated text
                    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    generated_part = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
                    
                    result = f"Full: \"{full_text}\"\n"
                    result += f"Generated: \"{generated_part}\"\n"
                    print(result.strip())
                    interp_file.write(result)
                    
                    # Analyze symbolic consistency
                    consistency_note = analyze_symbolic_consistency(prompt, generated_part)
                    if consistency_note:
                        print(f"Symbolic analysis: {consistency_note}")
                        interp_file.write(f"Symbolic analysis: {consistency_note}\n")
                    
                except Exception as e:
                    error_msg = f"Error during generation: {e}\n"
                    print(error_msg.strip())
                    interp_file.write(error_msg)
            
            interp_file.write("\n" + "-"*80 + "\n")
    
    print(f"Symbolic interpretability test results saved to: {interp_log_path}")


def analyze_symbolic_consistency(prompt, generated_text):
    """
    Simple analysis of whether the generated text maintains symbolic consistency.
    This is a basic heuristic analysis - in practice, you'd want more sophisticated
    symbolic reasoning evaluation.
    """
    prompt_lower = prompt.lower()
    generated_lower = generated_text.lower()
    
    # Check for logical completion patterns
    if "if" in prompt_lower and "then" in prompt_lower:
        if any(word in generated_lower for word in ["then", "therefore", "so"]):
            return "Logical structure maintained"
    
    # Check for sequence continuation
    if any(seq in prompt_lower for seq in ["a b c", "1 2 3", "monday tuesday"]):
        return "Sequence pattern detected"
    
    # Check for mathematical patterns
    if any(math_word in prompt_lower for math_word in ["equals", "+", "-", "plus", "minus"]):
        if any(math_word in generated_lower for math_word in ["=", "equals", "is"]):
            return "Mathematical reasoning maintained"
    
    return None


def create_symbolic_config(args):
    """Create a configuration specifically for the symbolic transformer."""
    # Start with base preset
    config = get_preset_config(args.preset)
    
    # Set model type to symbolic
    config.model_type = "Symbolic"
    config.transformer_block_type = "Symbolic"
    
    # Override with command line arguments
    if args.block_size is not None:
        config.block_size = args.block_size
    if args.max_position_embeddings is not None:
        config.max_position_embeddings = args.max_position_embeddings
    if args.n_layer is not None:
        config.n_layer = args.n_layer
    if args.n_head is not None:
        config.n_head = args.n_head
    if args.n_embd is not None:
        config.n_embd = args.n_embd
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.bias is not None:
        config.bias = args.bias
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    # Symbolic-specific parameters
    config.use_symbolic_ffn = args.use_symbolic_ffn
    config.use_vocab_refinement = args.use_vocab_refinement
    config.use_v = args.use_v
    config.use_proj = args.use_proj
    
    # Training parameters
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    config.generation_max_len = args.generation_max_len
    config.temperature = args.temperature
    config.top_k = args.top_k
    
    # Re-run post_init to validate
    config.__post_init__()
    
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Pure Symbolic Transformer with ALiBi')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use")
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large', 'character'],
                       help='Model size preset')
    parser.add_argument('--block_size', type=int, default=None,
                       help='Training sequence length')
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                       help='Maximum sequence length for inference')
    parser.add_argument("--n_layer", type=int, default=None, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=None, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability")
    parser.add_argument("--bias", action='store_true', help="Use bias in linear layers")
    
    # Symbolic-specific parameters
    parser.add_argument("--use_symbolic_ffn", action='store_true', default=True,
                       help="Use vocabulary-constrained FFN (default: True)")
    parser.add_argument("--no_symbolic_ffn", action='store_false', dest='use_symbolic_ffn',
                       help="Disable symbolic FFN")
    parser.add_argument("--use_vocab_refinement", action='store_true', default=False,
                       help="Use vocabulary refinement in projections")
    parser.add_argument("--use_v", action='store_true', default=False,
                       help="Use value projection in attention")
    parser.add_argument("--use_proj", action='store_true', default=False,
                       help="Use output projection in attention")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, 
                       help="Max norm for gradient clipping")
    parser.add_argument("--trainer_type", type=str, default="simple",
                       help="Type of trainer to use")
    
    # Data and tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'],
                       help='Type of tokenizer to use')
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to a pretrained tokenizer directory")
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./output_symbolic',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=256,
                       help='Logging interval')
    parser.add_argument("--save_model_filename", type=str, default="symbolic_model.pt",
                       help="Filename for the saved model checkpoint")
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    # Generation testing
    parser.add_argument('--test_generation', action='store_true',
                       help='Test symbolic interpretability after training')
    parser.add_argument("--skip_generation", action="store_true", 
                       help="Skip sample text generation after training")
    parser.add_argument("--generation_max_len", type=int, default=30, 
                       help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.5, 
                       help="Sampling temperature (lower for symbolic reasoning)")
    parser.add_argument("--top_k", type=int, default=20, 
                       help="Top-k sampling parameter")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger, tee_output, logs_dir, timestamp = setup_logging(args.output_dir)
    
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        logger.info("="*60)
        logger.info("STARTING SYMBOLIC TRANSFORMER TRAINING")
        logger.info("="*60)
        logger.info(f"Session timestamp: {timestamp}")
        
        # Set device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        logger.info(f"Using device: {device}")
        
        # Create symbolic configuration
        config = create_symbolic_config(args)
        
        # Initialize tokenizer
        logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
        if args.tokenizer_path:
            tokenizer = create_tokenizer(args.tokenizer_type, from_pretrained=args.tokenizer_path)
        else:
            if args.tokenizer_type == "character":
                # Build character vocab from dataset
                temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
                temp_dataset_args = [args.dataset]
                if args.dataset_config:
                    temp_dataset_args.append(args.dataset_config)
                
                temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
                
                # Find text field
                if 'text' in temp_dataset.column_names:
                    text_samples = temp_dataset['text']
                elif 'story' in temp_dataset.column_names:
                    text_samples = temp_dataset['story']
                else:
                    text_field = next((col for col in temp_dataset.column_names 
                                     if temp_dataset.features[col].dtype == 'string'), None)
                    if not text_field:
                        logger.error(f"Could not find text column. Available: {temp_dataset.column_names}")
                        sys.exit(1)
                    text_samples = temp_dataset[text_field]
                
                tokenizer = create_tokenizer(args.tokenizer_type)
                tokenizer.build_vocab_from_texts([str(item) for item in text_samples])
            else:
                tokenizer = create_tokenizer(args.tokenizer_type)
        
        # Update config with tokenizer info
        config.update_from_tokenizer(tokenizer)
        
        # Print configuration with symbolic emphasis
        print("=" * 60)
        print("SYMBOLIC TRANSFORMER CONFIGURATION")
        print("=" * 60)
        print(f"Model Type:                Symbolic Transformer")
        print(f"Symbolic Constraints:      All internal states vocabulary-constrained")
        print(f"Channel Structure:         Preserved through symbolic layer norm")
        print(f"FFN Constraints:           Vocabulary-grounded projections")
        print(f"Use Symbolic FFN:          {config.use_symbolic_ffn}")
        print(f"Use Vocab Refinement:      {config.use_vocab_refinement}")
        print(f"Use V Projection:          {config.use_v}")
        print(f"Use Output Projection:     {config.use_proj}")
        print_config(config, dataset_name=args.dataset)
        if args.dataset_config:
            print(f"Dataset Config:          {args.dataset_config}")
        print(f"Max Samples:             {args.max_samples}")
        print("=" * 60)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        dataloader, tokenizer = load_and_prepare_data(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            max_seq_length=config.block_size,
            batch_size=config.batch_size,
            mlm=False,
            split='train',
            shuffle=True
        )
        logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")
        
        # Initialize symbolic model
        logger.info("Initializing Symbolic Transformer model...")
        model = get_model("Symbolic", config=config).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Symbolic Transformer initialized with {num_params/1e6:.2f}M parameters.")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize trainer
        trainer = get_trainer(
            trainer_type=args.trainer_type,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_epochs=config.num_epochs,
            output_dir=args.output_dir,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval
        )
        
        # Train the model
        logger.info("="*60)
        logger.info("STARTING SYMBOLIC TRAINING")
        logger.info("="*60)
        trainer.train()
        logger.info("SYMBOLIC TRAINING COMPLETED")
        
        # Save model
        model_path = os.path.join(args.output_dir, args.save_model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'tokenizer': tokenizer,
            'training_args': vars(args),
            'timestamp': timestamp,
        }, model_path)
        logger.info(f"Symbolic model saved to {model_path}")
        
        # Generate sample text
        if not args.skip_generation:
            logger.info("="*60)
            logger.info("TESTING SYMBOLIC GENERATION")
            logger.info("="*60)
            
            symbolic_prompts = [
                "The door was locked.  Tim had a key to the door.  Tim used ",
                "The brave knight",
                "Spotty loved the sun",
                "The bird was a shiny",
            ]
            
            model.eval()
            for i, prompt in enumerate(symbolic_prompts):
                logger.info(f"\nSymbolic test {i+1}: '{prompt}'")
                try:
                    _, generated_text = run_generation(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt,
                        device=device,
                        max_new_tokens=args.generation_max_len,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        show_progress=False
                    )
                    logger.info(f"Generated: {generated_text}")
                except Exception as e:
                    logger.error(f"Error generating for '{prompt}': {e}")
        
        # Test symbolic interpretability
        if args.test_generation:
            test_symbolic_interpretability(model, tokenizer, device, config, args.output_dir, timestamp)
        
        print("\n" + "="*60)
        print("SYMBOLIC TRANSFORMER TRAINING COMPLETED!")
        print("="*60)
        print(f"Key symbolic features demonstrated:")
        print(f"- All internal states remain vocabulary-interpretable")
        print(f"- Channel-wise normalization preserves head structure")
        print(f"- FFN outputs constrained to symbolic manifold")
        print(f"- {num_params/1e6:.2f}M parameters with full symbolic interpretability")
        print("="*60)
        
    finally:
        sys.stdout = original_stdout
        if hasattr(tee_output, 'close'):
            tee_output.close()


if __name__ == "__main__":
    main()
