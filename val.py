#!/usr/bin/env python
"""
Process validation checkpoints and create validation performance graphs.
Simple implementation based on check.py structure.
"""

import torch
import matplotlib.pyplot as plt
import json
import os
import glob
from pathlib import Path
import argparse

def extract_checkpoint_metrics(checkpoint_path):
    """Extract validation metrics from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        metrics = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': checkpoint.get('epoch', 0),
            'global_batch': checkpoint.get('global_batch', 0),
            'train_loss': checkpoint.get('loss', float('nan')),
            'val_loss': checkpoint.get('val_loss', float('nan')),
            'val_perplexity': checkpoint.get('val_perplexity', float('nan'))
        }
        
        return metrics
    except Exception as e:
        print(f"Error reading {checkpoint_path}: {e}")
        return None

def extract_jsonl_metrics(jsonl_path):
    """Extract validation metrics from JSONL log files."""
    metrics = []
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                # Extract validation events
                if data.get('event_type') == 'validation':
                    epoch = data.get('epoch', 0)
                    val_metrics = data.get('metrics', {})
                    
                    metrics.append({
                        'epoch': epoch,
                        'global_batch': val_metrics.get('global_batch', 0),
                        'val_loss': val_metrics.get('val_loss', float('nan')),
                        'val_perplexity': val_metrics.get('val_perplexity', float('nan'))
                    })
        
        return metrics
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return []

def create_validation_graphs(output_dir):
    """Create validation performance graphs from checkpoints and logs."""
    
    print(f"📊 Processing validation data from: {output_dir}")
    
    # Find checkpoint files
    checkpoint_patterns = [
        '*.pt',
        'checkpoint_*.pt', 
        'checkpoints/*.pt'
    ]
    
    checkpoint_metrics = []
    for pattern in checkpoint_patterns:
        checkpoint_files = list(Path(output_dir).glob(pattern))
        for cp_file in checkpoint_files:
            metrics = extract_checkpoint_metrics(str(cp_file))
            if metrics:
                checkpoint_metrics.append(metrics)
    
    # Find JSONL log files
    jsonl_files = list(Path(output_dir).glob('logs/*.jsonl'))
    jsonl_metrics = []
    for jsonl_file in jsonl_files:
        metrics = extract_jsonl_metrics(str(jsonl_file))
        jsonl_metrics.extend(metrics)
    
    print(f"Found {len(checkpoint_metrics)} checkpoint metrics")
    print(f"Found {len(jsonl_metrics)} JSONL validation records")
    
    # Create graphs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Validation Performance Analysis', fontsize=16)
    
    # Graph 1: Validation Loss over Epochs (from checkpoints)
    if checkpoint_metrics:
        epochs = [m['epoch'] for m in checkpoint_metrics if not pd.isna(m['val_loss'])]
        val_losses = [m['val_loss'] for m in checkpoint_metrics if not pd.isna(m['val_loss'])]
        
        if epochs and val_losses:
            axes[0,0].plot(epochs, val_losses, 'bo-', label='Validation Loss')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Validation Loss')
            axes[0,0].set_title('Validation Loss vs Epoch (Checkpoints)')
            axes[0,0].grid(True)
            axes[0,0].legend()
    
    # Graph 2: Validation Perplexity over Epochs (from checkpoints)
    if checkpoint_metrics:
        epochs = [m['epoch'] for m in checkpoint_metrics if not pd.isna(m['val_perplexity'])]
        val_ppls = [m['val_perplexity'] for m in checkpoint_metrics if not pd.isna(m['val_perplexity'])]
        
        if epochs and val_ppls:
            axes[0,1].plot(epochs, val_ppls, 'ro-', label='Validation Perplexity')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Validation Perplexity')
            axes[0,1].set_title('Validation Perplexity vs Epoch (Checkpoints)')
            axes[0,1].grid(True)
            axes[0,1].legend()
    
    # Graph 3: Validation Loss over Global Batches (from JSONL)
    if jsonl_metrics:
        batches = [m['global_batch'] for m in jsonl_metrics if not pd.isna(m['val_loss'])]
        val_losses = [m['val_loss'] for m in jsonl_metrics if not pd.isna(m['val_loss'])]
        
        if batches and val_losses:
            axes[1,0].plot(batches, val_losses, 'go-', label='Validation Loss', alpha=0.7)
            axes[1,0].set_xlabel('Global Batch')
            axes[1,0].set_ylabel('Validation Loss')
            axes[1,0].set_title('Validation Loss vs Global Batch (Training Log)')
            axes[1,0].grid(True)
            axes[1,0].legend()
    
    # Graph 4: Validation Perplexity over Global Batches (from JSONL)
    if jsonl_metrics:
        batches = [m['global_batch'] for m in jsonl_metrics if not pd.isna(m['val_perplexity'])]
        val_ppls = [m['val_perplexity'] for m in jsonl_metrics if not pd.isna(m['val_perplexity'])]
        
        if batches and val_ppls:
            axes[1,1].plot(batches, val_ppls, 'mo-', label='Validation Perplexity', alpha=0.7)
            axes[1,1].set_xlabel('Global Batch') 
            axes[1,1].set_ylabel('Validation Perplexity')
            axes[1,1].set_title('Validation Perplexity vs Global Batch (Training Log)')
            axes[1,1].grid(True)
            axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save graph
    graph_path = os.path.join(output_dir, 'validation_analysis.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"📈 Saved validation graph: {graph_path}")
    
    # Print summary
    print(f"\n📋 VALIDATION SUMMARY:")
    if checkpoint_metrics:
        sorted_checkpoints = sorted(checkpoint_metrics, key=lambda x: x.get('val_loss', float('inf')))
        best = sorted_checkpoints[0] if sorted_checkpoints else None
        
        if best:
            print(f"🏆 Best checkpoint: {best['checkpoint']}")
            print(f"   Epoch: {best['epoch']}")
            print(f"   Val Loss: {best['val_loss']:.4f}")
            print(f"   Val Perplexity: {best['val_perplexity']:.2f}")
    
    if jsonl_metrics:
        final_val = jsonl_metrics[-1] if jsonl_metrics else None
        if final_val:
            print(f"📍 Final validation:")
            print(f"   Epoch: {final_val['epoch']}")
            print(f"   Global Batch: {final_val['global_batch']}")
            print(f"   Val Loss: {final_val['val_loss']:.4f}")
            print(f"   Val Perplexity: {final_val['val_perplexity']:.2f}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Create validation performance graphs')
    parser.add_argument('--output_dir', type=str, default='./outputs/sym_4gpu_final/batch_metrics',
                       help='Directory containing checkpoints and logs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Directory not found: {args.output_dir}")
        return
    
    create_validation_graphs(args.output_dir)

if __name__ == "__main__":
    # Add pandas for nan checking
    try:
        import pandas as pd
    except ImportError:
        # Simple fallback for nan checking
        import math
        class pd:
            @staticmethod
            def isna(x):
                return x != x or math.isnan(x) if isinstance(x, (int, float)) else False
    
    main()