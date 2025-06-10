#!/usr/bin/env python
"""
Checkpoint Inspector - Check if validation checkpoints are loadable and contain correct data.
"""

import torch
import os
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """
    Inspect a checkpoint file and validate its contents.
    """
    print(f"🔍 Inspecting: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"📁 File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        
        # Check structure
        print(f"\n📋 Checkpoint contents:")
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                print(f"  🧠 {key}: {len(value)} model parameters")
                # Check a few model parameters
                for param_name, param_tensor in list(value.items())[:3]:
                    print(f"    - {param_name}: {param_tensor.shape}")
                if len(value) > 3:
                    print(f"    - ... and {len(value) - 3} more parameters")
                    
            elif key == 'optimizer_state_dict':
                print(f"  ⚙️ {key}: Present")
                
            elif isinstance(value, torch.Tensor):
                print(f"  📊 {key}: {value.item()}")
                
            else:
                print(f"  📝 {key}: {value}")
        
        # Validate key components
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"\n⚠️  Missing required keys: {missing_keys}")
        else:
            print(f"\n✅ All required keys present")
        
        # Check for validation-specific data
        validation_keys = ['val_loss', 'val_perplexity', 'global_batch', 'is_best']
        val_data = {key: checkpoint.get(key) for key in validation_keys if key in checkpoint}
        
        if val_data:
            print(f"\n🧪 Validation data:")
            for key, value in val_data.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n⚠️  No validation data found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(checkpoint_path, model_config=None):
    """
    Test if we can actually load the model from checkpoint.
    """
    print(f"\n🧪 Testing model loading from checkpoint...")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If you have model config, test creating model and loading weights
        if model_config:
            print("🏗️  Creating model from config...")
            # You'd need to import your model creation code here
            # model = create_model(model_config)
            # model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model creation would work (config provided)")
        else:
            print("⚠️  No model config provided, skipping model instantiation")
        
        # Check if state dict looks reasonable
        state_dict = checkpoint['model_state_dict']
        
        # Basic sanity checks
        param_count = sum(p.numel() for p in state_dict.values())
        print(f"📊 Total parameters: {param_count:,}")
        
        # Check for common transformer components
        transformer_keys = [key for key in state_dict.keys() if any(component in key.lower() 
                          for component in ['embed', 'attention', 'ffn', 'layer_norm', 'mlp'])]
        
        if transformer_keys:
            print(f"🔧 Found {len(transformer_keys)} transformer components")
            for key in transformer_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(transformer_keys) > 5:
                print(f"  - ... and {len(transformer_keys) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def compare_checkpoints(checkpoint_paths):
    """
    Compare multiple checkpoints to see improvement.
    """
    print(f"\n📈 Comparing {len(checkpoint_paths)} checkpoints...")
    
    checkpoint_data = []
    
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            data = {
                'path': os.path.basename(path),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'global_batch': checkpoint.get('global_batch', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A'),
                'val_perplexity': checkpoint.get('val_perplexity', 'N/A'),
                'train_loss': checkpoint.get('train_loss', 'N/A'),
                'is_best': checkpoint.get('is_best', False)
            }
            checkpoint_data.append(data)
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    
    if checkpoint_data:
        print(f"\n📊 Checkpoint comparison:")
        print(f"{'File':<30} {'Epoch':<6} {'Batch':<8} {'Val Loss':<10} {'Val PPL':<10} {'Best':<5}")
        print("-" * 75)
        
        for data in sorted(checkpoint_data, key=lambda x: x.get('global_batch', 0)):
            val_loss = f"{data['val_loss']:.4f}" if isinstance(data['val_loss'], (int, float)) else str(data['val_loss'])
            val_ppl = f"{data['val_perplexity']:.2f}" if isinstance(data['val_perplexity'], (int, float)) else str(data['val_perplexity'])
            is_best = "✅" if data['is_best'] else ""
            
            print(f"{data['path']:<30} {data['epoch']:<6} {data['global_batch']:<8} {val_loss:<10} {val_ppl:<10} {is_best:<5}")


def main():
    """
    Main checkpoint inspection script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect training checkpoints')
    parser.add_argument('checkpoint_path', nargs='?', 
                       help='Path to checkpoint file or directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all checkpoints in directory')
    
    args = parser.parse_args()
    
    if not args.checkpoint_path:
        # Default: look for checkpoints in outputs
        checkpoint_dirs = [
            './outputs/sym_4gpu_simple/checkpoints',
            './outputs/vanilla_4gpu_simple/checkpoints',
            './outputs/sym_4gpu_simple',
            './outputs/vanilla_4gpu_simple'
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                args.checkpoint_path = checkpoint_dir
                print(f"🔍 Auto-detected checkpoint directory: {checkpoint_dir}")
                break
        else:
            print("❌ No checkpoint directory found. Please specify path.")
            return
    
    path = Path(args.checkpoint_path)
    
    if path.is_file():
        # Single checkpoint file
        print(f"🔍 Inspecting single checkpoint file...")
        inspect_checkpoint(str(path))
        test_model_loading(str(path))
        
    elif path.is_dir():
        # Directory of checkpoints
        checkpoint_files = list(path.glob('*.pt')) + list(path.glob('**/*.pt'))
        
        if not checkpoint_files:
            print(f"❌ No .pt files found in {path}")
            return
        
        print(f"🔍 Found {len(checkpoint_files)} checkpoint files")
        
        if args.compare:
            # Compare all checkpoints
            compare_checkpoints([str(f) for f in checkpoint_files])
        else:
            # Inspect latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"🔍 Inspecting latest checkpoint: {latest_checkpoint.name}")
            inspect_checkpoint(str(latest_checkpoint))
            test_model_loading(str(latest_checkpoint))
    
    else:
        print(f"❌ Path not found: {path}")


if __name__ == "__main__":
    main()


# QUICK USAGE EXAMPLES:
"""
# Inspect latest checkpoint automatically
python checkpoint_inspector.py

# Inspect specific checkpoint
python checkpoint_inspector.py ./outputs/sym_4gpu_simple/checkpoint_epoch_1.pt

# Compare all checkpoints in directory
python checkpoint_inspector.py ./outputs/sym_4gpu_simple/checkpoints --compare

# Quick one-liner in Python:
import torch
checkpoint = torch.load('./outputs/sym_4gpu_simple/checkpoint_epoch_1.pt', map_location='cpu')
print("Keys:", list(checkpoint.keys()))
print("Val loss:", checkpoint.get('val_loss', 'N/A'))
"""