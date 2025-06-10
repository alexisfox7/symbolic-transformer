import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

def extract_training_data(log_dir, experiment_name, json_log_steps=50):
    """Extract training data from JSON logs."""
    json_files = []
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.endswith('.jsonl') and experiment_name in f:
                json_files.append(os.path.join(log_dir, f))
    
    print(f'Found {len(json_files)} JSON log files for {experiment_name}')
    
    all_batches = []
    all_losses = []
    all_perplexities = []
    epoch_boundaries = []
    batch_counter = 0
    
    for json_file in sorted(json_files):
        print(f'Processing: {json_file}')
        with open(json_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    
                    # Track training batches
                    if event.get('event_type') == 'batch':
                        loss = event.get('metrics', {}).get('loss')
                        perplexity = event.get('metrics', {}).get('perplexity')
                        if loss is not None:
                            batch_counter += json_log_steps
                            all_batches.append(batch_counter)
                            all_losses.append(loss)
                            all_perplexities.append(perplexity if perplexity is not None else 0)
                    
                    # Track epoch boundaries
                    elif event.get('event_type') == 'epoch_end':
                        epoch = event.get('epoch', 0)
                        if batch_counter > 0:
                            epoch_boundaries.append((batch_counter, epoch))
                
                except:
                    continue
    
    return all_batches, all_losses, all_perplexities, epoch_boundaries

def subsample_data(batches, values, every_n=5):
    """Take every nth point to reduce density"""
    return batches[::every_n], values[::every_n]

def create_comparison_plot(vanilla_data, symbolic_data):
    """Create comparison plot between vanilla and symbolic training."""
    
    vanilla_batches, vanilla_losses, vanilla_perplexities, vanilla_epochs = vanilla_data
    symbolic_batches, symbolic_losses, symbolic_perplexities, symbolic_epochs = symbolic_data
    purple = '#9467bd'
    yellow = '#ffc107'

        # Set paper-quality style
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 16,
        'figure.titlesize': 20
    })

    plt.figure(figsize=(16, 12))

    # Plot 1: Raw Training Loss Comparison (subsampled)
    plt.subplot(2, 2, 1)
    plt.gca().set_facecolor('#f5f5f5')
    if vanilla_batches and vanilla_losses:
        v_batches_sub, v_losses_sub = subsample_data(vanilla_batches, vanilla_losses, every_n=3)
        plt.plot(v_batches_sub, v_losses_sub, alpha=1.0, linewidth=1.5, 
                label='Vanilla Transformer', color=yellow)

    if symbolic_batches and symbolic_losses:
        s_batches_sub, s_losses_sub = subsample_data(symbolic_batches, symbolic_losses, every_n=3)
        plt.plot(s_batches_sub, s_losses_sub, alpha=1.0, linewidth=1.5, 
                label='Symbolic Transformer', color=purple)

    plt.title('Training Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha =0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    # Plot 2: Subsampled Training Loss
    plt.gca().set_facecolor('#f5f5f5')

    # Smooth vanilla (skip first 50 points)
    if len(vanilla_perplexities) > 50:
        window = min(10, (len(vanilla_perplexities) - 50) // 5)
        vanilla_smoothed = []
        vanilla_batches_smoothed = []
        
        for i in range(50, len(vanilla_perplexities)):
            start = max(50, i - window // 2)  # Don't go below index 50
            end = min(len(vanilla_perplexities), i + window // 2)
            vanilla_smoothed.append(sum(vanilla_perplexities[start:end]) / (end - start))
            vanilla_batches_smoothed.append(vanilla_batches[i])
        
        plt.plot(vanilla_batches_smoothed, vanilla_smoothed, linewidth=2, color=yellow, 
                label='Vanilla')

    # Smooth symbolic (skip first 50 points)
    if len(symbolic_perplexities) > 50:
        window = min(10, (len(symbolic_perplexities) - 50) // 5)
        symbolic_smoothed = []
        symbolic_batches_smoothed = []
        
        for i in range(50, len(symbolic_perplexities)):
            start = max(50, i - window // 2)  # Don't go below index 50
            end = min(len(symbolic_perplexities), i + window // 2)
            symbolic_smoothed.append(sum(symbolic_perplexities[start:end]) / (end - start))
            symbolic_batches_smoothed.append(symbolic_batches[i])
        
        plt.plot(symbolic_batches_smoothed, symbolic_smoothed, linewidth=2, color=purple, 
                label='Symbolic')
    plt.title('Perplexity Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    

    # Plot 3: Perplexity with subsampling
    plt.subplot(2, 2, 2)
    plt.gca().set_facecolor('#f5f5f5')
    if vanilla_batches and vanilla_perplexities:
        start_idx = int(0.05 * len(vanilla_perplexities))
        v_batches_sub, v_perps_sub = subsample_data(vanilla_batches[start_idx:], vanilla_perplexities[start_idx:], every_n=3)
        valid_vanilla = [(b, p) for b, p in zip(v_batches_sub, v_perps_sub) if p > 0]
        if valid_vanilla:
            v_batches, v_perps = zip(*valid_vanilla)
            plt.plot(v_batches, v_perps, alpha=1.0, linewidth=2, 
                    label='Vanilla Perplexity', color=yellow)

    if symbolic_batches and symbolic_perplexities:
        start_idx = int(0.05 * len(symbolic_perplexities))
        s_batches_sub, s_perps_sub = subsample_data(symbolic_batches[start_idx:], symbolic_perplexities[start_idx:], every_n=3)
        valid_symbolic = [(b, p) for b, p in zip(s_batches_sub, s_perps_sub) if p > 0]
        if valid_symbolic:
            s_batches, s_perps = zip(*valid_symbolic)
            plt.plot(s_batches, s_perps, alpha=1.0, linewidth=2, 
                    label='Symbolic Perplexity', color=purple)

    plt.title('Perplexity Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 4: Final Loss Progression
    plt.subplot(2, 2, 3)
#    Smooth vanilla
    plt.gca().set_facecolor('#f5f5f5')
    if len(vanilla_losses) > 50:
        window = min(10, len(symbolic_losses) // 5)
        vanilla_smoothed = []
        for i in range(len(vanilla_losses)):
            start = max(0, i - window // 2)
            end = min(len(vanilla_losses), i + window // 2)
            vanilla_smoothed.append(sum(vanilla_losses[start:end]) / (end - start))
        plt.plot(vanilla_batches, vanilla_smoothed, linewidth=2, color=yellow, 
                label='Vanilla')
    
    # Smooth symbolic
    if len(symbolic_losses) > 50:
        window = min(10, len(symbolic_losses) // 5)
        symbolic_smoothed = []
        for i in range(len(symbolic_losses)):
            start = max(0, i - window // 2)
            end = min(len(symbolic_losses), i + window // 2)
            symbolic_smoothed.append(sum(symbolic_losses[start:end]) / (end - start))
        plt.plot(symbolic_batches, symbolic_smoothed, linewidth=2, color=purple, 
                label='Symbolic')
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    

    plt.tight_layout()
    return plt

def print_comparison_summary(vanilla_data, symbolic_data):
    """Print comparison summary statistics."""
    vanilla_batches, vanilla_losses, vanilla_perplexities, vanilla_epochs = vanilla_data
    symbolic_batches, symbolic_losses, symbolic_perplexities, symbolic_epochs = symbolic_data
    
    print(f'\n=== VANILLA vs SYMBOLIC COMPARISON SUMMARY ===')
    
    if vanilla_losses:
        vanilla_valid_perps = [p for p in vanilla_perplexities if p > 0]
        print(f'Vanilla Transformer:')
        print(f'  Total training steps: {max(vanilla_batches)}')
        print(f'  Final loss: {vanilla_losses[-1]:.4f}')
        print(f'  Best loss: {min(vanilla_losses):.4f}')
        if vanilla_valid_perps:
            print(f'  Final perplexity: {vanilla_valid_perps[-1]:.2f}')
            print(f'  Best perplexity: {min(vanilla_valid_perps):.2f}')
        print(f'  Epochs: {max([e for _, e in vanilla_epochs]) if vanilla_epochs else 0}')
    
    if symbolic_losses:
        symbolic_valid_perps = [p for p in symbolic_perplexities if p > 0]
        print(f'Symbolic Transformer:')
        print(f'  Total training steps: {max(symbolic_batches)}')
        print(f'  Final loss: {symbolic_losses[-1]:.4f}')
        print(f'  Best loss: {min(symbolic_losses):.4f}')
        if symbolic_valid_perps:
            print(f'  Final perplexity: {symbolic_valid_perps[-1]:.2f}')
            print(f'  Best perplexity: {min(symbolic_valid_perps):.2f}')
        print(f'  Epochs: {max([e for _, e in symbolic_epochs]) if symbolic_epochs else 0}')
    
    if vanilla_losses and symbolic_losses:
        final_loss_diff = symbolic_losses[-1] - vanilla_losses[-1]
        best_loss_diff = min(symbolic_losses) - min(vanilla_losses)
        print(f'\nDifferences (Symbolic - Vanilla):')
        print(f'  Final loss difference: {final_loss_diff:.4f}')
        print(f'  Best loss difference: {best_loss_diff:.4f}')
        
        # Perplexity comparison
        vanilla_valid_perps = [p for p in vanilla_perplexities if p > 0]
        symbolic_valid_perps = [p for p in symbolic_perplexities if p > 0]
        if vanilla_valid_perps and symbolic_valid_perps:
            final_perp_diff = symbolic_valid_perps[-1] - vanilla_valid_perps[-1]
            best_perp_diff = min(symbolic_valid_perps) - min(vanilla_valid_perps)
            print(f'  Final perplexity difference: {final_perp_diff:.2f}')
            print(f'  Best perplexity difference: {best_perp_diff:.2f}')
        
        if final_loss_diff < 0:
            print(f'  → Symbolic achieved {abs(final_loss_diff):.4f} lower final loss')
        else:
            print(f'  → Vanilla achieved {final_loss_diff:.4f} lower final loss')

def main():
    # Configuration
    vanilla_log_dir = './outputs/vanilla_4gpu_final/logs'
    vanilla_experiment = 'vanilla_4gpu_final'
    
    symbolic_log_dir = './outputs/sym_4gpu_final/logs'
    symbolic_experiment = 'symbolic_4gpu_final'
    
    json_log_steps = 50
    
    print('=== VANILLA vs SYMBOLIC TRAINING COMPARISON ===')
    print(f'JSON logging frequency: every {json_log_steps} steps\n')
    
    # Extract data from both models
    print('Extracting Vanilla data...')
    vanilla_data = extract_training_data(vanilla_log_dir, vanilla_experiment, json_log_steps)
    
    print('\nExtracting Symbolic data...')
    symbolic_data = extract_training_data(symbolic_log_dir, symbolic_experiment, json_log_steps)
    
    # Create comparison plot (now includes perplexity)
    print('\nCreating comparison plot with perplexity...')
    plt = create_comparison_plot(vanilla_data, symbolic_data)
    
    # Save plot
    output_path = './outputs/vanilla_vs_symbolic_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Comparison plot saved to: {output_path}')
    plt.close()
    
    # Print summary
    print_comparison_summary(vanilla_data, symbolic_data)
    
    print(f'\n=== COMPARISON COMPLETE ===')

if __name__ == "__main__":
    main()