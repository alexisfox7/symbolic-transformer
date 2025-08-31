#!/usr/bin/env python3
"""
Analyze individual Wikipedia article token lengths.
"""

import sys
import os
sys.path.append('/Users/alexisfox/st/src')

from datasets import load_dataset
from mytokenizers import GPT2Tokenizer
from itertools import islice
from tqdm import tqdm
import numpy as np
import json

def main():
    # Load English Wikipedia with streaming
    print("Loading Wikipedia dataset (streaming mode)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

    # How many articles to analyze
    NUM_ARTICLES = 50000  # Manageable sample for quick analysis

    # Initialize tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer()

    # Collect article lengths
    print(f"Analyzing token lengths for {NUM_ARTICLES} articles...")
    lengths = []
    
    for i, article in enumerate(tqdm(islice(ds['train'], NUM_ARTICLES), total=NUM_ARTICLES, desc="Processing articles")):
        text = article['text']
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
        
        # Print progress every 10k articles
        if (i + 1) % 10000 == 0:
            current_avg = np.mean(lengths)
            print(f"Progress: {i+1:,} articles processed, current avg: {current_avg:.1f} tokens")

    # Convert to numpy array for analysis
    lengths = np.array(lengths)
    
    # Calculate statistics
    stats = {
        'total_articles': len(lengths),
        'mean_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'percentiles': {
            '5th': float(np.percentile(lengths, 5)),
            '10th': float(np.percentile(lengths, 10)),
            '25th': float(np.percentile(lengths, 25)),
            '75th': float(np.percentile(lengths, 75)),
            '90th': float(np.percentile(lengths, 90)),
            '95th': float(np.percentile(lengths, 95)),
            '99th': float(np.percentile(lengths, 99))
        }
    }
    
    # Count articles by length ranges
    length_bins = {
        'very_short_0_50': np.sum(lengths <= 50),
        'short_51_200': np.sum((lengths > 50) & (lengths <= 200)),
        'medium_201_512': np.sum((lengths > 200) & (lengths <= 512)),
        'long_513_1000': np.sum((lengths > 512) & (lengths <= 1000)),
        'very_long_1001_2000': np.sum((lengths > 1000) & (lengths <= 2000)),
        'extremely_long_2000_plus': np.sum(lengths > 2000)
    }
    
    stats['length_distribution'] = length_bins
    
    # Print results
    print(f"\n=== Wikipedia Article Length Analysis ===")
    print(f"Sample size: {stats['total_articles']:,} articles")
    print(f"Mean length: {stats['mean_length']:.1f} tokens")
    print(f"Median length: {stats['median_length']:.1f} tokens")
    print(f"Std deviation: {stats['std_length']:.1f} tokens")
    print(f"Min length: {stats['min_length']} tokens")
    print(f"Max length: {stats['max_length']:,} tokens")
    
    print(f"\n=== Percentiles ===")
    for percentile, value in stats['percentiles'].items():
        print(f"{percentile}: {value:.1f} tokens")
    
    print(f"\n=== Length Distribution ===")
    total = stats['total_articles']
    for range_name, count in length_bins.items():
        percentage = (count / total) * 100
        range_display = range_name.replace('_', ' ').title()
        print(f"{range_display}: {count:,} articles ({percentage:.1f}%)")
    
    # Training implications
    print(f"\n=== Training Implications (block_size=512) ===")
    short_articles = np.sum(lengths <= 512)
    long_articles = np.sum(lengths > 512)
    
    # Estimate training examples
    short_tokens = np.sum(lengths[lengths <= 512])
    long_tokens = np.sum(lengths[lengths > 512])
    
    examples_from_short = short_tokens // 512  # Short articles combined
    examples_from_long = np.sum(lengths[lengths > 512] // 512)  # Long articles split
    
    total_estimated_examples = examples_from_short + examples_from_long
    
    print(f"Articles ≤ 512 tokens: {short_articles:,} ({short_articles/total*100:.1f}%)")
    print(f"Articles > 512 tokens: {long_articles:,} ({long_articles/total*100:.1f}%)")
    print(f"Estimated training examples: {total_estimated_examples:,}")
    print(f"Articles per training example: {total/total_estimated_examples:.2f}")
    
    # Save detailed results
    output_file = "wikipedia_length_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()