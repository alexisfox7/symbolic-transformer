#!/usr/bin/env python
"""
Process .pt validation checkpoints and create validation performance graphs.
Runs validation on each checkpoint to compute metrics.
"""

import torch
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import argparse
import math
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import get_preset_config
    from mytokenizers import create_tokenizer
    from model import get_model
    from utils.data_utils import load_and_prepare_data
    from torch.utils.data import DataLoader, random_split
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def is_nan(x):
    """Simple NaN check."""
    if isinstance(x, (int, float)):
        return math.isnan(x) or x != x
    return False

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

def infer_config_from_checkpoint(checkpoint_path, base_config):
    """Infer model configuration from checkpoint dimensions."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Find model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        clean_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            clean_state_dict[new_key] = value
        
        # Create a copy of the base config to avoid modifying the original
        import copy
        config = copy.deepcopy(base_config)
        
        # Get dimensions from attention weights (most reliable)
        for key, tensor in clean_state_dict.items():
            if 'attn.c_attn.weight' in key:
                # c_attn projects to 3 * d_model (q, k, v)
                out_features, in_features = tensor.shape
                d_model = in_features  # input features = d_model
                
                config.d_model = d_model
                config.d_ff = 4 * d_model  # Standard transformer ratio
                
                # Infer n_head (usually d_model // 64, but check if it divides evenly)
                for head_size in [64, 32, 128, 16]:
                    if d_model % head_size == 0:
                        config.n_head = d_model // head_size
                        break
                
                print(f"  Detected d_model={d_model}, n_head={config.n_head}")
                break
        
        # Get vocab size from embedding layer
        if 'transformer.wte.weight' in clean_state_dict:
            vocab_size, _ = clean_state_dict['transformer.wte.weight'].shape
            config.vocab_size = vocab_size
            print(f"  Detected vocab_size={vocab_size}")
        
        # Get number of layers by counting
        layer_count = 0
        for key in clean_state_dict.keys():
            if 'transformer.h.' in key:
                layer_num = int(key.split('transformer.h.')[1].split('.')[0])
                layer_count = max(layer_count, layer_num + 1)
        
        if layer_count > 0:
            config.n_layer = layer_count
            print(f"  Detected n_layer={layer_count}")
        
        print(f"  Final config: d_model={config.d_model}, n_layer={config.n_layer}, n_head={config.n_head}, vocab_size={config.vocab_size}")
        
        return config
        
    except Exception as e:
        print(f"  Warning: Could not infer config from checkpoint: {e}")
        return base_config

def evaluate_checkpoint(checkpoint_path, val_dataloader, base_config, device='cuda'):
    """Evaluate a single checkpoint on validation data."""
    print(f"🧪 Evaluating: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Find model state dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 0)
            global_batch = checkpoint.get('global_batch', 0)
            train_loss = checkpoint.get('loss', float('nan'))
        else:
            # Checkpoint IS the model state dict
            model_state_dict = checkpoint
            epoch = 0
            global_batch = 0
            train_loss = float('nan')
        
        # Simple dimension detection from embedding weights
        model_config = base_config
        if 'transformer.wte.weight' in model_state_dict:
            vocab_size, d_model = model_state_dict['transformer.wte.weight'].shape
            model_config.vocab_size = vocab_size
            model_config.d_model = d_model
            model_config.d_ff = 4 * d_model
            model_config.n_head = max(1, d_model // 64)
            print(f"  Using d_model={d_model} from checkpoint")
        elif 'module.transformer.wte.weight' in model_state_dict:
            vocab_size, d_model = model_state_dict['module.transformer.wte.weight'].shape
            model_config.vocab_size = vocab_size
            model_config.d_model = d_model
            model_config.d_ff = 4 * d_model
            model_config.n_head = max(1, d_model // 64)
            print(f"  Using d_model={d_model} from checkpoint")
        
        # Create model
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
                # Remove 'module.' prefix from checkpoint keys
                fixed_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    fixed_state_dict[new_key] = value
                model.load_state_dict(fixed_state_dict)
            else:
                print(f"❌ Error loading model: {e}")
                return None
        
        model.eval()
        
        # Run validation
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for val_batch_data in val_dataloader:
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
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        results = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': epoch,
            'global_batch': global_batch,
            'train_loss': train_loss,
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_samples': total_samples
        }
        
        print(f"  Val Loss: {avg_loss:.4f}, Val PPL: {perplexity:.2f}")
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating checkpoint: {e}")
        return None

def create_validation_graphs(output_dir, dataset_name, max_samples, preset, batch_size, device):
    """Create validation performance graphs from .pt checkpoint files."""
    
    print(f"📊 Processing .pt files from: {output_dir}")
    
    # Setup model config and validation data
    config = get_preset_config(preset)
    tokenizer = create_tokenizer('gpt2')
    config.update_from_tokenizer(tokenizer)
    
    val_dataloader, tokenizer = load_validation_data(
        dataset_name, tokenizer, max_samples, 
        config.block_size, batch_size
    )
    
    # Find .pt checkpoint files
    checkpoint_patterns = ['*.pt', 'checkpoint_*.pt', 'checkpoints/*.pt']
    
    checkpoint_metrics = []
    for pattern in checkpoint_patterns:
        checkpoint_files = list(Path(output_dir).glob(pattern))
        for cp_file in sorted(checkpoint_files):
            print(f"Processing: {cp_file.name}")
            metrics = evaluate_checkpoint(str(cp_file), val_dataloader, config, device)
            if metrics:
                checkpoint_metrics.append(metrics)
    
    print(f"Found {len(checkpoint_metrics)} checkpoint files with validation data")
    
    if not checkpoint_metrics:
        print("❌ No validation data found in checkpoint files")
        return
    
    # Create graphs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Validation Performance Analysis', fontsize=16)
    
    # Graph 1: Validation Loss over Epochs
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    
    if epochs and val_losses:
        axes[0,0].plot(epochs, val_losses, 'bo-', label='Validation Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Validation Loss')
        axes[0,0].set_title('Validation Loss vs Epoch')
        axes[0,0].grid(True)
        axes[0,0].legend()
    
    # Graph 2: Validation Perplexity over Epochs
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['val_perplexity'])]
    val_ppls = [m['val_perplexity'] for m in checkpoint_metrics if not is_nan(m['val_perplexity'])]
    
    if epochs and val_ppls:
        axes[0,1].plot(epochs, val_ppls, 'ro-', label='Validation Perplexity')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Validation Perplexity')
        axes[0,1].set_title('Validation Perplexity vs Epoch')
        axes[0,1].grid(True)
        axes[0,1].legend()
    
    # Graph 3: Validation Loss over Global Batches
    batches = [m['global_batch'] for m in checkpoint_metrics if not is_nan(m['val_loss']) and m['global_batch'] > 0]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss']) and m['global_batch'] > 0]
    
    if batches and val_losses:
        axes[1,0].plot(batches, val_losses, 'go-', label='Validation Loss')
        axes[1,0].set_xlabel('Global Batch')
        axes[1,0].set_ylabel('Validation Loss')
        axes[1,0].set_title('Validation Loss vs Global Batch')
        axes[1,0].grid(True)
        axes[1,0].legend()
    
    # Graph 4: Train vs Validation Loss
    train_losses = [m['train_loss'] for m in checkpoint_metrics if not is_nan(m['train_loss'])]
    val_losses = [m['val_loss'] for m in checkpoint_metrics if not is_nan(m['val_loss'])]
    epochs = [m['epoch'] for m in checkpoint_metrics if not is_nan(m['train_loss']) and not is_nan(m['val_loss'])]
    
    if epochs and train_losses and val_losses:
        axes[1,1].plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
        axes[1,1].plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_title('Train vs Validation Loss')
        axes[1,1].grid(True)
        axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save graph
    graph_path = os.path.join(output_dir, 'validation_analysis.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved validation graph: {graph_path}")
    
    # Print summary
    print(f"\n📋 VALIDATION SUMMARY:")
    sorted_checkpoints = sorted(checkpoint_metrics, key=lambda x: x.get('val_loss', float('inf')))
    best = sorted_checkpoints[0] if sorted_checkpoints else None
    
    if best:
        print(f"🏆 Best checkpoint: {best['checkpoint']}")
        print(f"   Epoch: {best['epoch']}")
        print(f"   Val Loss: {best['val_loss']:.4f}")
        print(f"   Val Perplexity: {best['val_perplexity']:.2f}")
    
    print(f"\n📊 All checkpoints:")
    for i, cp in enumerate(sorted_checkpoints[:5]):  # Show top 5
        marker = "🏆" if i == 0 else f"{i+1}."
        print(f"{marker} {cp['checkpoint']} - Val Loss: {cp['val_loss']:.4f}, PPL: {cp['val_perplexity']:.2f}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create validation performance graphs')
    parser.add_argument('--output_dir', type=str, default='./outputs/vanilla_4gpu_final/batch_metrics',
                       help='Directory containing checkpoints')
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
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Directory not found: {args.output_dir}")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    create_validation_graphs(
        args.output_dir, args.dataset, args.max_samples, 
        args.preset, args.batch_size, device
    )

if __name__ == "__main__":
    main()