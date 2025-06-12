#!/usr/bin/env python
"""
Evaluate validation performance on saved checkpoints.
"""

import torch
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import get_preset_config
from mytokenizers import create_tokenizer
from modelold import get_model
from utils.data_utils import load_and_prepare_data
from torch.utils.data import DataLoader, random_split
import math
from tqdm import tqdm

def load_validation_data(dataset_name, tokenizer, max_samples, block_size, batch_size, val_ratio=0.1):
    """Load and prepare validation data."""
    print(f"📊 Loading validation data...")
    
    # Load full dataset
    full_dataloader, tokenizer = load_and_prepare_data(
        dataset_name=dataset_name,
        dataset_config="",
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_seq_length=block_size,
        batch_size=batch_size,
        mlm=False,
        split='train',
        shuffle=False
    )
    
    # Create train/val split
    full_dataset = full_dataloader.dataset
    generator = torch.Generator().manual_seed(42)
    
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        collate_fn=full_dataloader.collate_fn,
        drop_last=False,
        num_workers=0
    )
    
    print(f"✅ Validation data: {len(val_dataloader)} batches, {len(val_dataset)} samples")
    return val_dataloader, tokenizer

def evaluate_checkpoint(checkpoint_path, val_dataloader, model_config, device='cuda'):
    """
    Evaluate a single checkpoint on validation data.
    """
    print(f"\n🧪 Evaluating: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # DEBUG: Check what keys are in checkpoint
        print(f"🔍 Checkpoint keys: {len(list(checkpoint.keys()))} total keys")
        
        # Check if checkpoint IS the model state dict (keys start with 'module.')
        first_key = list(checkpoint.keys())[0] if checkpoint else ""
        
        if first_key.startswith('module.'):
            # Checkpoint IS the model state dict
            model_state_dict = checkpoint
            print(f"✅ Checkpoint is model state dict directly")
            
            # DETECT MODEL CONFIG FROM CHECKPOINT
            print(f"🔧 Auto-detecting model config from checkpoint...")
            
            # Get embedding dimension from checkpoint
            wte_key = None
            for key in model_state_dict.keys():
                if 'wte.weight' in key or 'token_embedding' in key:
                    wte_key = key
                    break
            
            if wte_key:
                n_embd = model_state_dict[wte_key].shape[1]
                vocab_size = model_state_dict[wte_key].shape[0]
                print(f"📊 Detected n_embd: {n_embd}, vocab_size: {vocab_size}")
                
                # Update model config to match checkpoint
                model_config.n_embd = n_embd
                model_config.vocab_size = vocab_size
                
                # Update head dimension if needed
                if hasattr(model_config, 'n_head') and model_config.n_head > 0:
                    model_config.head_dim = n_embd // model_config.n_head
                    print(f"📊 Updated head_dim: {model_config.head_dim}")
            
            # Extract metadata if available (probably not in this format)
            epoch = 'N/A'
            global_batch = 'N/A' 
            train_loss = 'N/A'
            
        else:
            # Normal checkpoint format
            model_state_key = None
            possible_keys = ['model_state_dict', 'model', 'state_dict', 'net']
            
            for key in possible_keys:
                if key in checkpoint:
                    model_state_key = key
                    break
            
            if model_state_key is None:
                print(f"❌ No model state dict found. Available keys: {list(checkpoint.keys())}")
                return None
            
            print(f"✅ Found model state at key: '{model_state_key}'")
            model_state_dict = checkpoint[model_state_key]
            
            # Get checkpoint metadata
            epoch = checkpoint.get('epoch', 'N/A')
            global_batch = checkpoint.get('global_batch', 'N/A')
            train_loss = checkpoint.get('loss', 'N/A')
        
        # Create model with corrected config
        is_symbolic = ('symbolic' in checkpoint_path.lower() or 
                      getattr(model_config, 'use_symbolic_ffn', False))
        
        if is_symbolic:
            model = get_model("Symbolic", config=model_config).to(device)
        else:
            model = get_model("Vanilla", config=model_config).to(device)
        
        # Load model weights with key fixing
        try:
            model.load_state_dict(model_state_dict)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "module." in str(e):
                print(f"🔧 Fixing checkpoint keys (removing 'module.' prefix)...")
                
                # Remove 'module.' prefix from checkpoint keys
                fixed_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    fixed_state_dict[new_key] = value
                
                print(f"✅ Fixed {len(fixed_state_dict)} keys")
                model.load_state_dict(fixed_state_dict)
            else:
                print(f"❌ Model loading error: {e}")
                print(f"💡 This might be a config mismatch. Check model dimensions.")
                raise e
        
        model.eval()
        
        print(f"✅ Model loaded from checkpoint")
        
        # Get checkpoint metadata (moved after model loading)
        # epoch, global_batch, train_loss already set above
        
        # Run validation
        total_loss = 0.0
        total_samples = 0
        val_batches = 0
        
        print(f"🔄 Running validation...")
        
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)
            
            for val_batch_data in progress_bar:
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in val_batch_data.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.get('loss')
                
                if loss is not None and not torch.isnan(loss):
                    batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    val_batches += 1
                    
                    # Update progress
                    current_avg = total_loss / total_samples
                    progress_bar.set_postfix({'val_loss': f'{current_avg:.4f}'})
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        # Results
        results = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': epoch,
            'global_batch': global_batch,
            'train_loss': train_loss,
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_samples': total_samples,
            'val_batches': val_batches
        }
        
        print(f"📊 Results:")
        print(f"  Validation Loss: {avg_loss:.4f}")
        print(f"  Validation Perplexity: {perplexity:.2f}")
        print(f"  Samples: {total_samples}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_checkpoints(checkpoint_paths, val_dataloader, model_config, device='cuda'):
    """
    Evaluate multiple checkpoints and compare results.
    """
    print(f"🏁 Comparing {len(checkpoint_paths)} checkpoints...")
    
    all_results = []
    
    for checkpoint_path in checkpoint_paths:
        result = evaluate_checkpoint(checkpoint_path, val_dataloader, model_config, device)
        if result:
            all_results.append(result)
    
    if all_results:
        print(f"\n📈 COMPARISON RESULTS:")
        print(f"{'Checkpoint':<30} {'Epoch':<6} {'Batch':<8} {'Train Loss':<12} {'Val Loss':<10} {'Val PPL':<10}")
        print("-" * 85)
        
        # Sort by validation loss
        sorted_results = sorted(all_results, key=lambda x: x.get('val_loss', float('inf')))
        
        for i, result in enumerate(sorted_results):
            train_loss = f"{result['train_loss']:.4f}" if isinstance(result['train_loss'], (int, float)) else str(result['train_loss'])
            val_loss = f"{result['val_loss']:.4f}" if isinstance(result['val_loss'], (int, float)) else str(result['val_loss'])
            val_ppl = f"{result['val_perplexity']:.2f}" if isinstance(result['val_perplexity'], (int, float)) else str(result['val_perplexity'])
            
            best_marker = "🏆" if i == 0 else "  "
            
            print(f"{best_marker} {result['checkpoint']:<28} {result['epoch']:<6} {result['global_batch']:<8} {train_loss:<12} {val_loss:<10} {val_ppl:<10}")
        
        # Best model
        best_result = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best_result['checkpoint']}")
        print(f"   Val Loss: {best_result['val_loss']:.4f}")
        print(f"   Val Perplexity: {best_result['val_perplexity']:.2f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on validation data')
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/sym_4gpu_simple',
                       help='Directory containing checkpoints')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Specific checkpoint file to evaluate')
    parser.add_argument('--dataset', type=str, default='roneneldan/TinyStories',
                       help='Dataset for validation')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Max samples for validation')
    parser.add_argument('--preset', type=str, default='small',
                       help='Model preset')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Validation batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Create model config
    config = get_preset_config(args.preset)
    
    # Create tokenizer
    tokenizer = create_tokenizer('gpt2')
    
    # Load validation data
    val_dataloader, tokenizer = load_validation_data(
        args.dataset, tokenizer, args.max_samples, 
        config.block_size, args.batch_size
    )
    
    # Update config with tokenizer
    config.update_from_tokenizer(tokenizer)
    
    # Find checkpoints
    if args.checkpoint_file:
        # Single checkpoint
        checkpoint_paths = [args.checkpoint_file]
    else:
        # All checkpoints in directory
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_paths = []
        
        # Look for epoch checkpoints
        for pattern in ['checkpoint_epoch_*.pt', 'checkpoints/*.pt', '*.pt']:
            checkpoint_paths.extend(list(checkpoint_dir.glob(pattern)))
        
        checkpoint_paths = [str(p) for p in checkpoint_paths if p.is_file()]
    
    if not checkpoint_paths:
        print(f"❌ No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"🔍 Found {len(checkpoint_paths)} checkpoints")
    
    # Evaluate
    if len(checkpoint_paths) == 1:
        # Single checkpoint
        evaluate_checkpoint(checkpoint_paths[0], val_dataloader, config, device)
    else:
        # Multiple checkpoints
        compare_checkpoints(checkpoint_paths, val_dataloader, config, device)

if __name__ == "__main__":
    main()

# QUICK USAGE EXAMPLES:
"""
# Evaluate all checkpoints in directory
python checkpoint_evaluator.py --checkpoint_dir ./outputs/sym_4gpu_simple

# Evaluate specific checkpoint
python checkpoint_evaluator.py --checkpoint_file ./outputs/sym_4gpu_simple/checkpoint_epoch_1.pt

# Use different dataset/settings
python checkpoint_evaluator.py --checkpoint_dir ./outputs/sym_4gpu_simple --max_samples 5000 --batch_size 8
"""