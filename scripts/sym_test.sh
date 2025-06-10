#!/bin/bash
# Enhanced Progressive Symbolic Transformer Training with Parameter Analysis
# Includes detailed parameter breakdown and analysis

set -e  # Exit on any error

# Configuration - matching original script
DIR="./outputs/sym_4gpu_analysis"
N=110000
EXPERIMENT_NAME="symbolic_4gpu_analysis"

# Model configuration - matching original
N_EMBD=384
PRESET="small"

# Multi-GPU configuration - matching original
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Simplified batch configuration (no gradient accumulation, no stages)
BATCH_SIZE=4  # Direct batch size per GPU

# JSON logging configuration
JSON_LOG_STEPS=64

echo "========================================================"
echo "SYMBOLIC TRANSFORMER 4-GPU TRAINING WITH PARAMETER ANALYSIS"
echo "========================================================"
echo "Output directory: $DIR"
echo "Max samples: $N"
echo "Number of GPUs: $NUM_GPUS"
echo "Model size: $N_EMBD dimensions"
echo "JSON logging: Every $JSON_LOG_STEPS batches"
echo "Experiment name: $EXPERIMENT_NAME"
echo ""
echo "Batch size: $BATCH_SIZE per GPU (${BATCH_SIZE}×4 = $((BATCH_SIZE * 4)) total)"
echo "========================================================"

# Create output directory and analysis subdirectory
mkdir -p $DIR
mkdir -p $DIR/analysis

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available or Python/PyTorch not working"
    exit 1
fi

# Create parameter analysis utility inline
echo "Creating parameter analysis utility..."
cat > $DIR/analysis/param_analyzer.py << 'EOF'
#!/usr/bin/env python
"""
Parameter analysis utility for the training run.
"""

import torch
from collections import defaultdict
import sys
import os

def analyze_model_parameters(model, model_name="Model", save_path=None):
    """Analyze model parameters with component breakdown."""
    
    # Handle accelerator wrapping
    original_model = model
    if hasattr(model, 'module'):
        print(f"✓ Model is accelerator-wrapped")
        model = model.module
        wrapper_info = "Accelerator-wrapped"
    else:
        wrapper_info = "Direct model"
    
    print(f"\n{'='*70}")
    print(f"PARAMETER ANALYSIS: {model_name}")
    print(f"Model Type: {wrapper_info}")
    print(f"{'='*70}")
    
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
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024*1024):.2f} MB")
    print(f"Parameter Efficiency: {total_params / 1e6:.2f}M params")
    
    # Component breakdown
    print(f"\nDETAILED COMPONENT BREAKDOWN:")
    print(f"{'-'*70}")
    
    sorted_components = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
    
    for component, count in sorted_components:
        percentage = (count / total_params) * 100
        size_mb = count * 4 / (1024 * 1024)
        print(f"{component:<35} {count:>10,} ({percentage:>5.1f}%) {size_mb:>8.2f} MB")
    
    # Layer summary
    if layer_counts:
        print(f"\nLAYER SUMMARY:")
        print(f"{'-'*50}")
        
        # Group layers by type
        embedding_params = layer_counts.get('Embeddings', 0)
        final_params = layer_counts.get('Final', 0)
        output_params = layer_counts.get('Output', 0)
        vocab_params = layer_counts.get('Vocab Grounding', 0)
        
        if embedding_params:
            print(f"{'Embeddings':<20} {embedding_params:>10,} ({embedding_params/total_params*100:>5.1f}%)")
        
        # Transformer layers
        layer_nums = []
        for layer_name in layer_counts:
            if layer_name.startswith('Layer '):
                layer_nums.append(int(layer_name.split()[1]))
        
        if layer_nums:
            layer_nums.sort()
            layer_total = sum(layer_counts[f'Layer {i}'] for i in layer_nums)
            avg_layer = layer_total / len(layer_nums)
            print(f"{'Transformer Layers':<20} {layer_total:>10,} ({layer_total/total_params*100:>5.1f}%)")
            print(f"{'  - Layers':<20} {len(layer_nums):>10} layers")
            print(f"{'  - Avg per layer':<20} {avg_layer:>10,.0f} params")
        
        if vocab_params:
            print(f"{'Vocab Grounding':<20} {vocab_params:>10,} ({vocab_params/total_params*100:>5.1f}%)")
        if final_params:
            print(f"{'Final LayerNorm':<20} {final_params:>10,} ({final_params/total_params*100:>5.1f}%)")
        if output_params:
            print(f"{'Output Head':<20} {output_params:>10,} ({output_params/total_params*100:>5.1f}%)")
    
    print(f"{'='*70}")
    
    # Save detailed analysis if requested
    if save_path:
        import json
        
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
        
        with open(save_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"\nDetailed analysis saved to: {save_path}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'components': dict(component_counts),
        'layers': dict(layer_counts)
    }

def analyze_trainer_model(trainer, save_dir=None):
    """Analyze model from accelerate trainer with full context."""
    
    print("ACCELERATE TRAINER ANALYSIS")
    print("=" * 50)
    
    # Accelerator information
    if hasattr(trainer, 'accelerator'):
        acc = trainer.accelerator
        print(f"Accelerator Device: {acc.device}")
        print(f"Mixed Precision: {acc.mixed_precision}")
        print(f"Number of Processes: {acc.num_processes}")
        print(f"Is Main Process: {acc.is_main_process}")
        print(f"Process Index: {acc.process_index}")
        
        # Memory estimation
        if hasattr(acc, 'state') and hasattr(acc.state, 'total_memory'):
            print(f"Available Memory: {acc.state.total_memory}")
    
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
        
        print(f"\nMEMORY ESTIMATES:")
        print(f"{'-'*40}")
        print(f"Model Memory: {model_memory_mb:.1f} MB per process")
        print(f"Gradient Memory: {gradient_memory_mb:.1f} MB per process") 
        print(f"Optimizer Memory: {optimizer_memory_mb:.1f} MB per process")
        print(f"Total Training Memory: {total_memory_per_process:.1f} MB per process")
        print(f"Total Across {trainer.accelerator.num_processes} Processes: {total_memory_all_processes:.1f} MB")
        print(f"Peak Memory per GPU: ~{total_memory_per_process * 1.5:.1f} MB (with overhead)")
    
    return results

if __name__ == "__main__":
    print("Parameter analysis utility ready.")
    print("Use: analyze_trainer_model(trainer, save_dir) or analyze_model_parameters(model)")
EOF

# Make the parameter analyzer executable
chmod +x $DIR/analysis/param_analyzer.py

# Training run with parameter analysis integration
echo ""
echo "========================================================"
echo "STARTING TRAINING WITH PARAMETER ANALYSIS"
echo "========================================================"

accelerate launch \
    --config_file ./accelerate_config_4gpu.yaml \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    examples/train_symbolic_with_param_analysis.py \
    --use_proj --use_v \
    --preset $PRESET \
    --n_embd $N_EMBD \
    --batch_size $BATCH_SIZE \
    --num_epochs 8 \
    --max_samples $N \
    --output_dir $DIR \
    --trainer_type accelerate \
    --json_log_steps $JSON_LOG_STEPS \
    --experiment_name $EXPERIMENT_NAME \
    --learning_rate 3e-4 \
    --clip_grad_norm 1.0 \
    --log_interval 32 \
    --analysis_dir $DIR/analysis

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"

# Post-training analysis
echo ""
echo "========================================================"
echo "POST-TRAINING PARAMETER ANALYSIS"
echo "========================================================"

python -c "
import sys
sys.path.append('$DIR/analysis')
from param_analyzer import analyze_model_parameters
import torch
import os

# Load the final model
model_path = '$DIR/symbolic_model.pt'
if os.path.exists(model_path):
    print('Loading final model for analysis...')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f'Model Configuration:')
        print(f'  Layers: {config.n_layer}')
        print(f'  Heads: {config.n_head}') 
        print(f'  Embedding Dim: {config.n_embd}')
        print(f'  Vocab Size: {config.vocab_size}')
        print(f'  Block Size: {config.block_size}')
        print(f'  Symbolic FFN: {getattr(config, \"use_symbolic_ffn\", \"Unknown\")}')
        print(f'  Use Proj: {getattr(config, \"use_proj\", \"Unknown\")}')
        print(f'  Use V: {getattr(config, \"use_v\", \"Unknown\")}')
    
    # Create model for analysis
    try:
        sys.path.append('.')
        from model import get_model
        
        model = get_model('Symbolic', config=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Full analysis
        results = analyze_model_parameters(
            model, 
            'Final Symbolic Transformer',
            save_path='$DIR/analysis/final_model_analysis.json'
        )
        
        # Training results summary
        if 'training_result' in checkpoint:
            training_result = checkpoint['training_result']
            print(f'\\nTRAINING RESULTS:')
            print(f'{\"=\"*40}')
            if 'final_loss' in training_result:
                print(f'Final Training Loss: {training_result[\"final_loss\"]:.6f}')
            if 'training_time' in training_result:
                print(f'Training Time: {training_result[\"training_time\"]:.1f} seconds')
            if 'epoch_losses' in training_result:
                losses = training_result['epoch_losses']
                print(f'Loss Progression: {losses[0]:.4f} -> {losses[-1]:.4f}')
                print(f'Loss Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%')
        
    except Exception as e:
        print(f'Could not create model for analysis: {e}')
        print('Using checkpoint state dict directly...')
        
        # Direct state dict analysis
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        
        print(f'\\nDIRECT CHECKPOINT ANALYSIS:')
        print(f'Total Parameters: {total_params:,}')
        print(f'Model Size: {total_params * 4 / (1024*1024):.2f} MB')
        
        # Component breakdown
        components = {}
        for name, param in state_dict.items():
            if 'transformer.wte' in name:
                comp = 'Token Embeddings'
            elif 'transformer.h.' in name and 'attn' in name:
                comp = 'Attention Layers'
            elif 'transformer.h.' in name and 'ffn' in name:
                comp = 'FFN Layers'
            elif 'lm_head' in name:
                comp = 'LM Head'
            elif 'vocab_grounding' in name:
                comp = 'Vocab Grounding'
            else:
                comp = 'Other'
            
            components[comp] = components.get(comp, 0) + param.numel()
        
        print(f'\\nComponent Breakdown:')
        for comp, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
            print(f'  {comp}: {count:,} ({count/total_params*100:.1f}%)')

else:
    print('No final model found for analysis.')
"

# Generate training plots with parameter annotations
echo ""
echo "========================================================"
echo "GENERATING ENHANCED TRAINING PLOTS"
echo "========================================================"

python -c "
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Enhanced plotting with parameter info
log_dir = '$DIR/logs'
analysis_dir = '$DIR/analysis'
json_files = []

if os.path.exists(log_dir):
    for f in os.listdir(log_dir):
        if f.endswith('.jsonl') and '$EXPERIMENT_NAME' in f:
            json_files.append(os.path.join(log_dir, f))

print(f'Found {len(json_files)} JSON log files')

# Load parameter analysis if available
param_info = {}
param_file = os.path.join(analysis_dir, 'parameter_analysis.json')
if os.path.exists(param_file):
    with open(param_file, 'r') as f:
        param_info = json.load(f)
    print(f'Loaded parameter analysis from {param_file}')

# Extract training metrics
all_batches = []
all_losses = []
epoch_boundaries = []

for json_file in sorted(json_files):
    print(f'Processing: {json_file}')
    with open(json_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                
                if event.get('event_type') == 'batch':
                    step = event.get('step', 0)
                    loss = event.get('metrics', {}).get('loss')
                    if loss is not None:
                        all_batches.append(step)
                        all_losses.append(loss)
                        
                elif event.get('event_type') == 'epoch_end':
                    epoch = event.get('epoch', 0)
                    global_batch = event.get('metrics', {}).get('global_batch', 0)
                    if global_batch > 0:
                        epoch_boundaries.append((global_batch, epoch))
            except:
                continue

if all_batches and all_losses:
    # Create enhanced training plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Symbolic Transformer Training Analysis with Parameter Breakdown', fontsize=16, fontweight='bold')
    
    # Plot 1: Training loss
    ax1 = axes[0, 0]
    ax1.plot(all_batches, all_losses, alpha=0.7, linewidth=1, label='Training Loss', color='blue')
    ax1.set_title('Training Loss vs Batches')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add epoch boundaries
    for batch, epoch in epoch_boundaries:
        ax1.axvline(x=batch, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(batch, max(all_losses) * 0.9, f'E{epoch}', rotation=90, verticalalignment='bottom')
    
    # Plot 2: Smoothed loss
    ax2 = axes[0, 1]
    if len(all_losses) > 50:
        window = min(50, len(all_losses) // 5)
        smoothed = []
        for i in range(len(all_losses)):
            start = max(0, i - window // 2)
            end = min(len(all_losses), i + window // 2)
            smoothed.append(sum(all_losses[start:end]) / (end - start))
        ax2.plot(all_batches, smoothed, linewidth=2, color='orange', label='Smoothed Loss')
    else:
        ax2.plot(all_batches, all_losses, linewidth=2, label='Training Loss')
    
    ax2.set_title('Smoothed Training Loss')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Parameter breakdown (if available)
    ax3 = axes[1, 0]
    if param_info and 'component_breakdown' in param_info:
        components = param_info['component_breakdown']
        # Group small components
        major_components = {}
        other_total = 0
        for comp, count in components.items():
            if count > param_info['total_params'] * 0.05:  # >5%
                major_components[comp] = count
            else:
                other_total += count
        
        if other_total > 0:
            major_components['Other'] = other_total
        
        labels = list(major_components.keys())
        sizes = list(major_components.values())
        
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Parameter Distribution\\n(Total: {param_info[\"total_params\"]:,} params)')
    else:
        ax3.text(0.5, 0.5, 'Parameter analysis\\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Parameter Breakdown')
    
    # Plot 4: Training summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Summary statistics
    summary_text = f'TRAINING SUMMARY\\n'
    summary_text += f'{"="*20}\\n'
    summary_text += f'Total Batches: {max(all_batches) if all_batches else 0:,}\\n'
    summary_text += f'Final Loss: {all_losses[-1]:.4f}\\n' if all_losses else ''
    summary_text += f'Best Loss: {min(all_losses):.4f}\\n' if all_losses else ''
    summary_text += f'Epochs: {max([e for _, e in epoch_boundaries]) if epoch_boundaries else 0}\\n'
    
    if param_info:
        summary_text += f'\\nMODEL INFO\\n'
        summary_text += f'{"="*20}\\n'
        summary_text += f'Parameters: {param_info.get(\"total_params\", \"N/A\"):,}\\n'
        summary_text += f'Model Size: {param_info.get(\"model_size_mb\", 0):.1f} MB\\n'
        summary_text += f'Type: {param_info.get(\"model_type\", \"Unknown\")}\\n'
    
    # GPU info
    summary_text += f'\\nTRAINING CONFIG\\n'
    summary_text += f'{"="*20}\\n'
    summary_text += f'GPUs: $NUM_GPUS\\n'
    summary_text += f'Batch Size: $BATCH_SIZE per GPU\\n'
    summary_text += f'Total Batch: $((BATCH_SIZE * NUM_GPUS))\\n'
    summary_text += f'Model Dim: $N_EMBD\\n'
    summary_text += f'Max Samples: $N\\n'
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plot_path = '$DIR/training_analysis_with_params.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Enhanced training plot saved to: {plot_path}')
    plt.close()
    
else:
    print('No valid training data found for plotting')
"

# Final summary
echo ""
echo "========================================================"
echo "SYMBOLIC TRANSFORMER TRAINING WITH ANALYSIS COMPLETED!"
echo "========================================================"
echo "Key features of this enhanced training:"
echo "  ✓ Comprehensive parameter analysis during training"
echo "  ✓ Component-wise parameter breakdown"
echo "  ✓ Memory usage estimation for multi-GPU setup"
echo "  ✓ Enhanced training plots with parameter information"
echo "  ✓ Detailed JSON logs with analysis integration"
echo ""
echo "Output files:"
echo "  Model: $DIR/symbolic_model.pt"
echo "  Logs: $DIR/logs/"
echo "  Analysis: $DIR/analysis/"
echo "  Plots: $DIR/training_analysis_with_params.png"
echo "  Parameter Data: $DIR/analysis/parameter_analysis.json"
echo ""
echo "The parameter analysis provides insights into:"
echo "  • Model size and memory requirements"
echo "  • Component-wise parameter distribution"
echo "  • Layer-by-layer breakdown"
echo "  • Training memory estimates"
echo "  • Accelerator-specific information"
echo "========================================================"