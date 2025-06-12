#!/usr/bin/env python
"""
FIXED Checkpoint Inspector - handles tensor values properly.
"""

import torch
import os

def safe_extract_value(value):
    """Safely extract value from tensor or scalar."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        else:
            return f"Tensor{list(value.shape)}"
    return value

def inspect_checkpoint_fixed(checkpoint_path):
    """
    FIXED: Inspect checkpoint without .item() errors.
    """
    print(f"🔍 Inspecting: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
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
                
            else:
                # FIXED: Handle both tensors and scalars
                safe_value = safe_extract_value(value)
                print(f"  📊 {key}: {safe_value}")
        
        # Validate key components
        required_keys = ['model_state_dict', 'optimizer_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"\n⚠️  Missing required keys: {missing_keys}")
        else:
            print(f"\n✅ All required keys present")
        
        # Check for validation-specific data
        validation_keys = ['val_loss', 'val_perplexity', 'global_batch', 'is_best', 'loss', 'epoch']
        print(f"\n🧪 Training/Validation data:")
        for key in validation_keys:
            if key in checkpoint:
                safe_value = safe_extract_value(checkpoint[key])
                print(f"  {key}: {safe_value}")
        
        # Quick validation check
        has_val_data = any(key.startswith('val_') for key in checkpoint.keys())
        if has_val_data:
            print(f"\n✅ Validation data found - checkpoint is from validation run")
        else:
            print(f"\n⚠️  No validation data - this might be a regular epoch checkpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

# Quick usage
if __name__ == "__main__":
    # Check your specific checkpoint
    checkpoint_paths = [
        "./outputs/sym_4gpu_simple/checkpoint_epoch_1.pt",
        "./outputs/sym_4gpu_simple/checkpoints/best_val_batch_000100.pt",  # If you have validation checkpoints
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"\n" + "="*60)
            inspect_checkpoint_fixed(path)
        else:
            print(f"\n❌ File not found: {path}")

# EVEN SIMPLER: Just check what's in the checkpoint
print(f"\n" + "="*60)
print("🔍 SIMPLE CHECKPOINT CHECK:")

try:
    checkpoint_path = "./outputs/sym_4gpu_simple/checkpoint_epoch_1.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"📋 Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Check specific values safely
    for key in ['epoch', 'loss', 'val_loss', 'val_perplexity', 'global_batch']:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"  {key}: {value.item()}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor{list(value.shape)}")
            else:
                print(f"  {key}: {value}")
    
    # Check model state dict size
    if 'model_state_dict' in checkpoint:
        model_params = checkpoint['model_state_dict']
        param_count = sum(p.numel() for p in model_params.values() if isinstance(p, torch.Tensor))
        print(f"  🧠 Model parameters: {param_count:,}")
    
    print(f"✅ Checkpoint is valid and loadable!")
    
except Exception as e:
    print(f"❌ Error: {e}")