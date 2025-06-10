#!/usr/bin/env python
# ./examples/train_symbolic_with_param_analysis.py
"""
Enhanced training script for Symbolic Transformer with comprehensive parameter analysis.
Includes detailed parameter breakdown, component analysis, and memory usage estimation.

Usage:
    python examples/train_symbolic_with_param_analysis.py --preset small --analysis_dir ./analysis
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime
from collections import defaultdict
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import SymbolicConfig, get_preset_config, print_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset

# JSON logging imports
from utils.json_logger import create_json_logger_for_training
from trainers.json_trainer import create_accelerate_trainer_with_json_logging

# Validation imports
from torch.utils.data import DataLoader, random_split

# Suppress output on non-main processes
if os.environ.get('LOCAL_RANK', '0') != '0': 
    sys.stdout = open(os.devnull, 'w')


def analyze_model_parameters(model, model_name="Model", save_path=None):
    """Comprehensive parameter analysis with component breakdown."""
    
    # Handle accelerator wrapping
    original_model = model
    if hasattr(model, 'module'):
        logger.info(f"✓ Model is accelerator-wrapped")
        model = model.module
        wrapper_info = "Accelerator-wrapped"
    else:
        wrapper_info = "Direct model"
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PARAMETER ANALYSIS: {model_name}")
    logger.info(f"Model Type: {wrapper_info}")
    logger.info(f"{'='*70}")
    
    # Component and layer tracking
    component_counts = defaultdict(int)
    layer_counts = defaultdict(int)
    param_details = []
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        size_mb = num_params * 4 / (1024 * 1024)
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        
        # Detailed component classification for symbolic transformer
        if 'transformer.wte' in name:
            component = 'Token Embeddings'
            layer = 'Embeddings'
        elif 'transformer.wpe' in name:
            component = 'Positional Embeddings'
            layer = 'Embeddings'
        elif 'transformer.ln_f' in name:
            component = 'Final LayerNorm'
            layer = 'Final'
        elif 'transformer.h.' in name:
            # Extract layer number
            parts = name.split('.')
            layer_idx = parts[2]
            layer = f'Layer {layer_idx}'
            
            if 'ln_1' in name:
                component = f'Layer {layer_idx} - Attention LayerNorm'
            elif 'ln_2' in name:
                component = f'Layer {layer_idx} - FFN LayerNorm'
            elif 'attn.c_attn' in name:
                component = f'Layer {layer_idx} - QKV Projection'
            elif 'attn.c_proj' in name:
                component = f'Layer {layer_idx} - Attention Output'
            elif 'attn.proj_tmp' in name:
                component = f'Layer {layer_idx} - Kronecker Projection'
            elif 'attn' in name:
                component = f'Layer {layer_idx} - Attention Other'
            elif 'ffn.channel_ffns' in name:
                component = f'Layer {layer_idx} - Channel FFN'
            elif 'ffn.channel_temperatures' in name:
                component = f'Layer {layer_idx} - FFN Temperature'
            elif 'ffn' in name:
                component = f'Layer {layer_idx} - FFN Other'
            else:
                component = f'Layer {layer_idx} - Other'
                
            layer_counts[layer] += num_params
        elif 'lm_head' in name:
            component = 'Language Model Head'
            layer = 'Output'
        elif 'vocab_grounding' in name:
            if 'channel_ffns' in name:
                component = 'Vocab Grounding - Channel FFN'
            elif 'channel_temperatures' in name:
                component = 'Vocab Grounding - Temperature'
            else:
                component = 'Vocab Grounding - Other'
            layer = 'Vocab Grounding'
        else:
            component = 'Other'
            layer = 'Other'
        
        component_counts[component] += num_params
        
        # Store parameter details
        param_details.append({
            'name': name,
            'component': component,
            'layer': layer,
            'shape': tuple(param.shape),
            'params': num_params,
            'size_mb': size_mb,
            'trainable': param.requires_grad
        })
    
    # Print summary
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    logger.info(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB")
    logger.info(f"Parameter Efficiency: {total_params / 1e6:.2f}M params")
    
    # Component breakdown
    logger.info(f"\nDETAILED COMPONENT BREAKDOWN:")
    logger.info(f"{'-'*70}")
    
    sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
    
    for component, count in sorted_components:
        percentage = (count / total_params) * 100
        size_mb = count * 4 / (1024 * 1024)
        logger.info(f"{component:<35} {count:>10,} ({percentage:>5.1f}%) {size_mb:>8.2f} MB")
    
    # Layer summary
    if layer_counts:
        logger.info(f"\nLAYER SUMMARY:")
        logger.info(f"{'-'*50}")
        
        # Group layers by type
        embedding_params = layer_counts.get('Embeddings', 0)
        final_params = layer_counts.get('Final', 0)
        output_params = layer_counts.get('Output', 0)
        vocab_params = layer_counts.get('Vocab Grounding', 0)
        
        if embedding_params:
            logger.info(f"{'Embeddings':<20} {embedding_params:>10,} ({embedding_params/total_params*100:>5.1f}%)")
        
        # Transformer layers
        layer_nums = []
        for layer_name in layer_counts:
            if layer_name.startswith('Layer '):
                layer_nums.append(int(layer_name.split()[1]))
        
        if layer_nums:
            layer_nums.sort()
            layer_total = sum(layer_counts[f'Layer {i}'] for i in layer_nums)
            avg_layer = layer_total / len(layer_nums)
            logger.info(f"{'Transformer Layers':<20} {layer_total:>10,} ({layer_total/total_params*100:>5.1f}%)")
            logger.info(f"{'  - Layers':<20} {len(layer_nums):>10} layers")
            logger.info(f"{'  - Avg per layer':<20} {avg_layer:>10,.0f} params")
        
        if vocab_params:
            logger.info(f"{'Vocab Grounding':<20} {vocab_params:>10,} ({vocab_params/total_params*100:>5.1f}%)")
        if final_params:
            logger.info(f"{'Final LayerNorm':<20} {final_params:>10,} ({final_params/total_params*100:>5.1f}%)")
        if output_params:
            logger.info(f"{'Output Head':<20} {output_params:>10,} ({output_params/total_params*100:>5.1f}%)")
    
    logger.info(f"{'='*70}")
    
    # Save detailed analysis if requested
    if save_path:
        analysis_data = {
            'model_name': model_name,
            'model_type': wrapper_info,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024*1024),
            'component_breakdown': dict(component_counts),
            'layer_breakdown': dict(layer_counts),
            'parameter_details': param_details
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"\nDetailed analysis saved to: {save_path}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'components': dict(component_counts),
        'layers': dict(layer_counts)
    }


def analyze_trainer_model(trainer, save_dir=None):
    """Analyze model from accelerate trainer with full context."""
    
    logger.info("ACCELERATE TRAINER ANALYSIS")
    logger.info("=" * 50)
    
    # Accelerator information
    if hasattr(trainer, 'accelerator'):
        acc = trainer.accelerator
        logger.info(f"Accelerator Device: {acc.device}")
        logger.info(f"Mixed Precision: {acc.mixed_precision}")
        logger.info(f"Number of Processes: {acc.num_processes}")
        logger.info(f"Is Main Process: {acc.is_main_process}")
        logger.info(f"Process Index: {acc.process_index}")
    
    # Model analysis
    model_class = trainer.model.__class__.__name__
    if hasattr(trainer.model, 'module'):
        model_class += " (Accelerator-wrapped)"
    
    save_path = None
    if save_dir:
        save_path = os.path.join(save_dir, 'parameter_analysis.json')
    
    results = analyze_model_parameters(trainer.model, f"Symbolic Transformer ({model_class})", save_path)
    
    # Training-specific memory estimates
    if hasattr(trainer, 'accelerator'):
        total_params = results['total_params']
        model_memory_mb = total_params * 4 / (1024 * 1024)  # FP32
        
        # Estimate training memory (model + gradients + optimizer states)
        gradient_memory_mb = model_memory_mb  # Same as model for FP32
        optimizer_memory_mb = model_memory_mb * 2  # AdamW: momentum + variance
        
        total_memory_per_process = model_memory_mb + gradient_memory_mb + optimizer_memory_mb
        total_memory_all_processes = total_memory_per_process * trainer.accelerator.num_processes
        
        logger.info(f"\nMEMORY ESTIMATES:")
        logger.info(f"{'-'*40}")
        logger.info(f"Model Memory: {model_memory_mb:.1f} MB per process")
        logger.info(f"Gradient Memory: {gradient_memory_mb:.1f} MB per process") 
        logger.info(f"Optimizer Memory: {optimizer_memory_mb:.1f} MB per process")
        logger.info(f"Total Training Memory: {total_memory_per_process:.1f} MB per process")
        logger.info(f"Total Across {trainer.accelerator.num_processes} Processes: {total_memory_all_processes:.1f} MB")
        logger.info(f"Peak Memory per GPU: ~{total_memory_per_process * 1.5:.1f} MB (with overhead)")
    
    return results


def parse_args():
    """Parse command line arguments with parameter analysis support."""
    parser = argparse.ArgumentParser(description='Train Symbolic Transformer with Parameter Analysis')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--block_size', type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--bias", action='store_true')
    
    # Symbolic-specific parameters
    parser.add_argument("--use_symbolic_ffn", action='store_true', default=True)
    parser.add_argument("--no_symbolic_ffn", action='store_false', dest='use_symbolic_ffn')
    parser.add_argument("--use_vocab_refinement", action='store_true', default=False)
    parser.add_argument("--use_v", action='store_true', default=False)
    parser.add_argument("--use_proj", action='store_true', default=False)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    
    # Trainer selection
    parser.add_argument("--trainer_type", type=str, default="accelerate",
                       choices=["simple", "accelerate"])
    
    # Checkpoint resumption
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # Tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'])
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./outputs/symbolic_analysis')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument("--save_model_filename", type=str, default="symbolic_model.pt")
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'])
    
    # Generation testing
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--generation_max_len", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=20)
    
    # JSON LOGGING ARGUMENTS
    parser.add_argument("--json_log_steps", type=int, default=100,
                       help="Log training metrics every N batches to JSON (default: 100)")
    parser.add_argument("--disable_json_logging", action="store_true",
                       help="Disable JSON logging")
    parser.add_argument("--experiment_name", type=str, default="symbolic_transformer",
                       help="Experiment name for JSON logs")
    
    # PARAMETER ANALYSIS ARGUMENTS (NEW)
    parser.add_argument("--analysis_dir", type=str, default=None,
                       help="Directory to save parameter analysis files")
    parser.add_argument("--skip_param_analysis", action="store_true",
                       help="Skip parameter analysis")
    
    return parser.parse_args()


def create_symbolic_config(args):
    """Create configuration for the symbolic transformer."""
    config = get_preset_config(args.preset)
    
    # Override with command line arguments
    if args.block_size is not None:
        config.block_size = args.block_size
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
    
    config.__post_init__()
    return config


def setup_logging_and_output(output_dir):
    """Setup logging and output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training function with comprehensive parameter analysis."""
    args = parse_args()

    # Setup
    logger = setup_logging_and_output(args.output_dir)
    
    # Setup analysis directory
    if args.analysis_dir is None:
        args.analysis_dir = os.path.join(args.output_dir, 'analysis')
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*80)
    logger.info("SYMBOLIC TRANSFORMER TRAINING WITH COMPREHENSIVE PARAMETER ANALYSIS")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Trainer: {args.trainer_type}")
    logger.info(f"Analysis Directory: {args.analysis_dir}")
    logger.info(f"Parameter Analysis: {'Enabled' if not args.skip_param_analysis else 'Disabled'}")
    
    # Create configuration
    config = create_symbolic_config(args)
    
    # Initialize tokenizer
    logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
    if args.tokenizer_type == "character":
        # Build character vocab from dataset sample
        temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
        temp_dataset_args = [args.dataset]
        if args.dataset_config:
            temp_dataset_args.append(args.dataset_config)
        
        temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
        
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
    
    # Print configuration
    print_config(config, dataset_name=args.dataset)
    
    # Setup JSON logging
    json_logger = None
    if not args.disable_json_logging:
        json_logger = create_json_logger_for_training(
            args.output_dir, 
            args.experiment_name, 
            args.json_log_steps
        )
        
        # Log initial configuration
        json_logger.log_config({
            'model_config': {
                'preset': args.preset,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'vocab_size': config.vocab_size,
                'block_size': config.block_size,
                'use_symbolic_ffn': config.use_symbolic_ffn,
                'use_v': config.use_v,
                'use_proj': config.use_proj,
            },
            'training_config': {
                'dataset': args.dataset,
                'max_samples': args.max_samples,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'learning_rate': config.learning_rate,
                'trainer_type': args.trainer_type,
            },
            'system_config': {
                'device': str(device),
                'tokenizer_type': args.tokenizer_type,
                'analysis_enabled': not args.skip_param_analysis,
            }
        })
        logger.info(f"JSON logging enabled: {json_logger.log_file}")
    
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
    
    # Initialize model
    logger.info("Initializing Symbolic Transformer...")
    model = get_model("Symbolic", config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {num_params/1e6:.2f}M parameters")
    
    # PARAMETER ANALYSIS BEFORE TRAINING
    if not args.skip_param_analysis:
        logger.info("\n" + "="*80)
        logger.info("PRE-TRAINING PARAMETER ANALYSIS")
        logger.info("="*80)
        
        pre_training_analysis = analyze_model_parameters(
            model, 
            "Pre-Training Symbolic Transformer",
            save_path=os.path.join(args.analysis_dir, 'pre_training_analysis.json')
        )
        
        # Log to JSON if available
        if json_logger:
            json_logger.log_custom("pre_training_analysis", {
                "total_params": pre_training_analysis['total_params'],
                "components": pre_training_analysis['components'],
                "layers": pre_training_analysis['layers']
            })
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create trainer with JSON logging
    logger.info(f"Setting up {args.trainer_type} trainer...")
    if args.trainer_type == "accelerate":
        trainer = create_accelerate_trainer_with_json_logging(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            json_logger=json_logger,
            num_epochs=config.num_epochs,
            output_dir=args.output_dir,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval
        )
    else:
        # Simple trainer fallback
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
    
    # PARAMETER ANALYSIS AFTER TRAINER SETUP
    if not args.skip_param_analysis:
        logger.info("\n" + "="*80)
        logger.info("POST-TRAINER-SETUP PARAMETER ANALYSIS")
        logger.info("="*80)
        
        trainer_analysis = analyze_trainer_model(trainer, save_dir=args.analysis_dir)
        
        # Log to JSON if available
        if json_logger:
            json_logger.log_custom("trainer_setup_analysis", {
                "trainer_type": args.trainer_type,
                "accelerator_info": {
                    "num_processes": getattr(trainer.accelerator, 'num_processes', 1) if hasattr(trainer, 'accelerator') else 1,
                    "mixed_precision": str(getattr(trainer.accelerator, 'mixed_precision', 'none')) if hasattr(trainer, 'accelerator') else 'none',
                    "device": str(getattr(trainer.accelerator, 'device', device)) if hasattr(trainer, 'accelerator') else str(device)
                },
                "parameter_analysis": trainer_analysis
            })
    
    # Train the model
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    training_result = trainer.train()
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    
    # POST-TRAINING PARAMETER ANALYSIS
    if not args.skip_param_analysis:
        logger.info("\n" + "="*80)
        logger.info("POST-TRAINING PARAMETER ANALYSIS")
        logger.info("="*80)
        
        post_training_analysis = analyze_model_parameters(
            trainer.model, 
            "Post-Training Symbolic Transformer",
            save_path=os.path.join(args.analysis_dir, 'post_training_analysis.json')
        )
        
        # Compare pre and post training
        if 'pre_training_analysis' in locals():
            logger.info(f"\nPARAMETER ANALYSIS COMPARISON:")
            logger.info(f"{'-'*50}")
            logger.info(f"Pre-training total params: {pre_training_analysis['total_params']:,}")
            logger.info(f"Post-training total params: {post_training_analysis['total_params']:,}")
            logger.info(f"Parameter change: {post_training_analysis['total_params'] - pre_training_analysis['total_params']:,}")
        
        # Log to JSON if available
        if json_logger:
            json_logger.log_custom("post_training_analysis", {
                "total_params": post_training_analysis['total_params'],
                "components": post_training_analysis['components'],
                "layers": post_training_analysis['layers'],
                "training_result": training_result
            })
    
    # Save final model with analysis
    model_path = os.path.join(args.output_dir, args.save_model_filename)
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.num_epochs,
        'config': config,
        'tokenizer': tokenizer,
        'training_args': vars(args),
        'training_result': training_result,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add parameter analysis to save dict
    if not args.skip_param_analysis and 'post_training_analysis' in locals():
        save_dict['parameter_analysis'] = post_training_analysis
    
    torch.save(save_dict, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Test generation
    if not args.skip_generation:
        logger.info("="*80)
        logger.info("TESTING SYMBOLIC GENERATION")
        logger.info("="*80)
        
        test_prompts = [
            "The brave knight",
            "Once upon a time",
            "Spotty the dog loved",
            "The door was locked. Tim had a key.",
        ]
        
        model.eval()
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nTest {i+1}: '{prompt}'")
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
                
                # Log generation to JSON
                if json_logger:
                    # Check if we're using accelerate trainer
                    is_main_process = True
                    if hasattr(trainer, 'accelerator'):
                        is_main_process = trainer.accelerator.is_main_process
                    elif hasattr(trainer, 'trainer') and hasattr(trainer.trainer, 'accelerator'):
                        is_main_process = trainer.trainer.accelerator.is_main_process
                    
                    if is_main_process:
                        json_logger.log_generation(
                            epoch=config.num_epochs,
                            prompt=prompt,
                            generated=generated_text,
                            generation_params={
                                'max_new_tokens': args.generation_max_len,
                                'temperature': args.temperature,
                                'top_k': args.top_k
                            }
                        )
                        
            except Exception as e:
                logger.error(f"Error generating for '{prompt}': {e}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SYMBOLIC TRANSFORMER TRAINING WITH PARAMETER ANALYSIS COMPLETED!")
    logger.info("="*80)
    logger.info(f"Model: {num_params/1e6:.2f}M parameters")
    logger.info(f"Final training loss: {training_result.get('final_loss', 'N/A')}")
    logger.info(f"Training time: {training_result.get('training_time', 'N/A')}")
    
    if not args.skip_param_analysis:
        logger.info(f"Parameter analysis files saved to: {args.analysis_dir}")
        logger.info("  - pre_training_analysis.json")
        logger.info("  - parameter_analysis.json (trainer setup)")
        logger.info("  - post_training_analysis.json")
    
    if json_logger:
        logger.info(f"JSON logs: {json_logger.log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()