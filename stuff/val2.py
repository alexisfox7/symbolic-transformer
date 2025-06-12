#!/usr/bin/env python3

import json
import sys

def fix_validation_entry(entry):
    """Fix a single validation log entry to match training format"""
    # Extract batch number from checkpoint file
    checkpoint = entry["checkpoint_file"]  # e.g., "model_batch_000100.pt"
    batch_num = int(checkpoint.split("_")[-1].replace(".pt", ""))
    
    # Build fixed entry matching training format
    fixed = {
        "timestamp": entry["timestamp"],
        "elapsed_time": entry["elapsed_time"],
        "event_type": "validation",
        "experiment_name": entry["experiment_name"],
        "epoch": 1,  # Match training epoch
        "batch": batch_num,
        "step": batch_num,
        "metrics": {
            "loss": entry["metrics"]["val_loss"],
            "perplexity": entry["metrics"]["val_perplexity"],
            "global_batch": batch_num,
            "samples": entry["metrics"]["val_samples"],
            "num_processes": 4
        }
    }
    return fixed

def main():
    input_file = "outputs/vanilla_4gpu_final/batch_metrics/validation_vanilla_4gpu_final_20250610_151331.jsonl"
    output_file = input_file.replace(".jsonl", "_fixed.jsonl")
    
    print(f"Reading: {input_file}")
    print(f"Writing: {output_file}")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
                fixed_entry = fix_validation_entry(entry)
                outfile.write(json.dumps(fixed_entry) + '\n')
                print(f"Fixed batch {fixed_entry['batch']}: loss={fixed_entry['metrics']['loss']:.4f}")
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    print(f"Done! Fixed validation log saved to: {output_file}")

if __name__ == "__main__":
    main()