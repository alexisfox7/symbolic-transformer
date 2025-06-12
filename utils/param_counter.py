#!/usr/bin/env python
# utils/param_counter.py
"""
Standalone parameter analysis tool for Symbolic Transformer models.
Simple but comprehensive parameter breakdown and analysis.

Usage:
    python utils/param_counter.py --model_path ./outputs/symbolic_model.pt
    python utils/param_counter.py --checkpoint ./outputs/checkpoint_epoch_5.pt
    python utils/param_counter.py --config_only --preset small --vocab_size 50257
"""

import argparse
import torch
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_model_parameters(model, model_name="Model"):
    """
    Analyze model parameters with clear component breakdown.
    
    Args:
        model: PyTorch model to analyze
        model_name: Name for display
        
    Returns:
        Dictionary with analysis results
    """
    
    # Handle accelerator wrapping
    original_model = model
    if hasattr(model, 'module'):
        print(f"✓ Model is accelerator-wrapped")
        model = model.module
        wrapper_type = "Accelerator-wrapped"
    else:
        wrapper_type = "Direct model"
    
    print(f"\n{'='*60}")
    print(f"PARAMETER ANALYSIS: {model_name}")
    print(f"Model Type: {wrapper_type}")
    print(f"{'='*60}")
    
    # Basic counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable: {total_params - trainable_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Parameter Efficiency: {total_params / 1e6:.2f}M")
    
    # Component analysis
    components = defaultdict(int)
    layers = defaultdict(int)
    param_types = defaultdict(int)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        
        # Component classification (specific to Symbolic Transformer)
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
            # Extract layer number: transformer.h.{layer}.{component}
            parts = name.split('.')
            layer_idx = parts[2] if len(parts) > 2 else 'unknown'
            layer = f'Layer {layer_idx}'
            
            if 'ln_1' in name:
                component = 'Attention LayerNorm'
            elif 'ln_2' in name:
                component = 'FFN LayerNorm'
            elif 'attn.c_attn' in name:
                component = 'QKV Projection'
            elif 'attn.c_proj' in name:
                component = 'Attention Output'
            elif 'attn.proj_tmp' in name:
                component = 'Kronecker Projection'
            elif 'attn' in name:
                component = 'Attention Other'
            elif 'ffn.channel_ffns' in name:
                component = 'Channel FFN'
            elif 'ffn.channel_temperatures' in name:
                component = 'FFN Temperature'
            elif 'ffn' in name:
                component = 'FFN Other'
            else:
                component = 'Layer Other'
                
            layers[layer] += num_params
        elif 'lm_head' in name:
            component = 'Language Model Head'
            layer = 'Output'
        elif 'vocab_grounding' in name:
            if 'channel_ffns' in name:
                component = 'Vocab Grounding FFN'
            elif 'channel_temperatures' in name:
                component = 'Vocab Grounding Temp'
            else:
                component = 'Vocab Grounding Other'
            layer = 'Vocab Grounding'
        else:
            component = 'Other'
            layer = 'Other'
        
        components[component] += num_params
        
        # Parameter type
        if 'weight' in name:
            param_types['Weights'] += num_params
        elif 'bias' in name:
            param_types['Biases'] += num_params
        elif 'temperature' in name:
            param_types['Temperatures'] += num_params
        else:
            param_types['Other'] += num_params
    
    # Print component breakdown
    print(f"\nCOMPONENT BREAKDOWN:")
    print(f"{'-'*60}")
    
    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
    for component, count in sorted_components:
        percentage = (count / total_params) * 100
        size_mb = count * 4 / (1024 * 1024)
        print(f"{component:<25} {count:>10,} ({percentage:>5.1f}%) {size_mb:>6.2f} MB")
    
    # Print layer summary
    if layers:
        print(f"\nLAYER SUMMARY:")
        print(f"{'-'*40}")
        
        # Sort layers by number
        layer_items = []
        other_items = []
        
        for layer_name, count in layers.items():
            if layer_name.startswith('Layer '):
                try:
                    layer_num = int(layer_name.split()[1])
                    layer_items.append((layer_num, layer_name, count))
                except:
                    other_items.append((layer_name, count))
            else:
                other_items.append((layer_name, count))
        
        # Print transformer layers first
        if layer_items:
            layer_items.sort()
            total_layer_params = sum(count for _, _, count in layer_items)
            avg_params = total_layer_params / len(layer_items)
            
            print(f"Transformer Layers ({len(layer_items)} layers):")
            print(f"  Total: {total_layer_params:,} ({total_layer_params/total_params*100:.1f}%)")
            print(f"  Average per layer: {avg_params:,.0f}")
            
            # Show individual layers if not too many
            if len(layer_items) <= 12:
                for _, layer_name, count in layer_items:
                    percentage = (count / total_params) * 100
                    print(f"  {layer_name}: {count:,} ({percentage:.1f}%)")
        
        # Print other components
        for layer_name, count in other_items:
            percentage = (count / total_params) * 100
            print(f"{layer_name}: {count:,} ({percentage:.1f}%)")
    
    # Print parameter types
    print(f"\nPARAMETER TYPES:")
    print(f"{'-'*30}")
    for param_type, count in sorted(param_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"{param_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"{'='*60}")
    
    return {
        'model_name': model_name,
        'model_type': wrapper_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'components': dict(components),
        'layers': dict(layers),
        'param_types': dict(param_types)
    }


def analyze_from_checkpoint(checkpoint_path):
    """Analyze model from checkpoint file."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Print checkpoint info
        print(f"\nCHECKPOINT INFO:")
        print(f"{'-'*30}")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.6f}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"Config: {config.n_layer}L-{config.n_head}H-{config.n_embd}D")
            print(f"Vocab Size: {config.vocab_size}")
            print(f"Block Size: {config.block_size}")
            if hasattr(config, 'use_symbolic_ffn'):
                print(f"Symbolic FFN: {config.use_symbolic_ffn}")
            if hasattr(config, 'use_proj'):
                print(f"Use Proj: {config.use_proj}")
            if hasattr(config, 'use_v'):
                print(f"Use V: {config.use_v}")
        
        # Try to create model for full analysis
        if 'config' in checkpoint and 'model_state_dict' in checkpoint:
            try:
                from modelold import get_model
                
                model = get_model("Symbolic", config=checkpoint['config'])
                model.load_state_dict(checkpoint['model_state_dict'])
                
                return analyze_model_parameters(model, f"Checkpoint Model (epoch {checkpoint.get('epoch', '?')})")
                
            except Exception as e:
                print(f"⚠️  Could not create model: {e}")
                print("Falling back to state dict analysis...")
        
        # Fallback: analyze state dict directly
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            
            print(f"\nSTATE DICT ANALYSIS:")
            print(f"{'-'*30}")
            print(f"Total Parameters: {total_params:,}")
            print(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB")
            
            # Component breakdown from state dict
            components = defaultdict(int)
            for name, param in state_dict.items():
                if 'transformer.wte' in name:
                    comp = 'Token Embeddings'
                elif 'transformer.h.' in name and 'attn' in name:
                    comp = 'Attention'
                elif 'transformer.h.' in name and 'ffn' in name:
                    comp = 'FFN'
                elif 'lm_head' in name:
                    comp = 'LM Head'
                elif 'vocab_grounding' in name:
                    comp = 'Vocab Grounding'
                else:
                    comp = 'Other'
                
                components[comp] += param.numel()
            
            print("\nComponent Breakdown:")
            for comp, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
                print(f"  {comp}: {count:,} ({count/total_params*100:.1f}%)")
            
            return {
                'total_params': total_params,
                'components': dict(components)
            }
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def analyze_from_config(preset, vocab_size, **kwargs):
    """Analyze model from config without training."""
    
    try:
        from config import get_preset_config
        from modelold import get_model
        
        config = get_preset_config(preset)
        config.vocab_size = vocab_size
        
        # Apply any additional config options
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.__post_init__()
        
        print(f"Creating model from config:")
        print(f"  Preset: {preset}")
        print(f"  Architecture: {config.n_layer}L-{config.n_head}H-{config.n_embd}D")
        print(f"  Vocab Size: {config.vocab_size}")
        print(f"  Block Size: {config.block_size}")
        
        model = get_model("Symbolic", config=config)
        
        return analyze_model_parameters(model, f"Symbolic Transformer ({preset})")
        
    except Exception as e:
        print(f"❌ Error creating model from config: {e}")
        return None


def save_analysis(analysis, output_path):
    """Save analysis results to JSON file."""
    
    if analysis is None:
        print("❌ No analysis to save")
        return
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"✅ Analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving analysis: {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Symbolic Transformer parameters')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model_path', type=str, 
                           help='Path to saved model (.pt file)')
    input_group.add_argument('--checkpoint', type=str,
                           help='Path to checkpoint file')
    input_group.add_argument('--config_only', action='store_true',
                           help='Analyze from config without loading model')
    
    # Config options (for --config_only)
    parser.add_argument('--preset', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model preset')
    parser.add_argument('--vocab_size', type=int, default=50257,
                       help='Vocabulary size')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, help='Embedding dimension')
    
    # Output options
    parser.add_argument('--save', type=str, help='Save analysis to JSON file')
    
    args = parser.parse_args()
    
    print("🔍 SYMBOLIC TRANSFORMER PARAMETER ANALYZER")
    print("=" * 50)
    
    # Run analysis based on input type
    analysis = None
    
    if args.model_path:
        analysis = analyze_from_checkpoint(args.model_path)
    elif args.checkpoint:
        analysis = analyze_from_checkpoint(args.checkpoint)
    elif args.config_only:
        config_kwargs = {}
        if args.n_layer: config_kwargs['n_layer'] = args.n_layer
        if args.n_head: config_kwargs['n_head'] = args.n_head
        if args.n_embd: config_kwargs['n_embd'] = args.n_embd
        
        analysis = analyze_from_config(args.preset, args.vocab_size, **config_kwargs)
    
    # Save analysis if requested
    if args.save and analysis:
        save_analysis(analysis, args.save)
    
    if analysis:
        print("\n✅ Analysis complete!")
    else:
        print("\n❌ Analysis failed!")


if __name__ == "__main__":
    main()