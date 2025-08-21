#!/usr/bin/env python3

from subtract import InteractiveTokenSandbox

def test_subtract_comparison():
    """Test the difference between raw and first_layer methods."""
    
    sandbox = InteractiveTokenSandbox(device='cuda' if True else 'cpu')
    
    # Set the same target as in your original example
    sandbox.set_target("The cat sat on the mat", layer=5, position=2)
    
    print("\n" + "="*80)
    print("COMPARING RAW vs FIRST LAYER METHODS")
    print("="*80)
    
    # Test with " cat" token using raw method
    print("\n--- RAW METHOD ---")
    sandbox.subtract(" cat", method='raw')
    sandbox.reset()
    
    # Test with " cat" token using first_layer method
    print("\n--- FIRST LAYER METHOD ---")
    sandbox.subtract(" cat", method='first_layer')
    sandbox.reset()
    
    # Compare multiple tokens
    print("\n" + "="*80)
    print("EXPERIMENT: Multiple tokens comparison")
    print("="*80)
    
    tokens_to_test = [" cat", " sat", " dog", " the", "cat"]
    
    print("\n--- RAW METHOD EXPERIMENT ---")
    results_raw = sandbox.experiment(tokens_to_test, method='raw')
    
    print("\n--- FIRST LAYER METHOD EXPERIMENT ---")
    results_first_layer = sandbox.experiment(tokens_to_test, method='first_layer')
    
    print("\n" + "="*60)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*60)
    print(f"{'Token':15s} {'Raw Scale':>10s} {'Raw Red%':>8s} {'FL Scale':>10s} {'FL Red%':>8s} {'Improvement':>12s}")
    print("-" * 75)
    
    for i, token in enumerate(tokens_to_test):
        raw_result = results_raw[i]
        fl_result = results_first_layer[i]
        
        improvement = fl_result['reduction_pct'] - raw_result['reduction_pct']
        
        print(f"{token:15s} {raw_result['scale']:10.3f} {raw_result['reduction_pct']:8.1f} "
              f"{fl_result['scale']:10.3f} {fl_result['reduction_pct']:8.1f} {improvement:12.1f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Calculate average improvements
    raw_avg = sum(r['reduction_pct'] for r in results_raw) / len(results_raw)
    fl_avg = sum(r['reduction_pct'] for r in results_first_layer) / len(results_first_layer)
    overall_improvement = fl_avg - raw_avg
    
    print(f"Average reduction with raw method: {raw_avg:.1f}%")
    print(f"Average reduction with first layer method: {fl_avg:.1f}%")
    print(f"Overall improvement: {overall_improvement:.1f} percentage points")
    
    if overall_improvement > 5:
        print("✓ First layer method shows significant improvement!")
    elif overall_improvement > 0:
        print("→ First layer method shows modest improvement")
    else:
        print("✗ First layer method doesn't improve performance")

if __name__ == "__main__":
    test_subtract_comparison()