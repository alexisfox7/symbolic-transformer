#!/usr/bin/env python3
"""
Analyze Wikipedia dataset using token_statistics module.
"""

import sys
import os
sys.path.append('/Users/alexisfox/st/src')

from datasets import load_dataset
from utils.token_statistics import TokenUsageAnalyzer
from mytokenizers import GPT2Tokenizer
from itertools import islice
from tqdm import tqdm

def main():
    # Load English Wikipedia with streaming
    print("Loading Wikipedia dataset (streaming mode)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)

    # How many articles to analyze
    NUM_ARTICLES = 10000

    # Collect texts with progress bar
    print(f"Collecting {NUM_ARTICLES} articles...")
    texts = []
    for article in tqdm(islice(ds['train'], NUM_ARTICLES), total=NUM_ARTICLES, desc="Collecting articles"):
        texts.append(article['text'])

    print(f"Collected {len(texts)} Wikipedia articles")

    # Initialize your tokenizer
    print("Initializing GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer()

    # Create analyzer
    analyzer = TokenUsageAnalyzer(tokenizer)

    # Run analysis
    print("Running token analysis...")
    results = analyzer.analyze_texts(
        texts=texts,
        max_samples=None,  # Use all collected texts
        store_sequences=False,  # Don't store sequences (saves memory)
        show_progress=True
    )

    # Print results
    print(f"\n=== Wikipedia EN Token Statistics ===")
    print(f"Total tokens: {results['total_tokens']:,}")
    print(f"Unique tokens: {results['unique_tokens']:,}")
    print(f"Number of samples: {results['num_samples']:,}")
    print(f"Average sequence length: {results['avg_sequence_length']:.1f}")
    print(f"Vocabulary coverage: {results['vocab_coverage']:.2%}")
    print(f"\nMost common tokens:")
    for i, (token, count) in enumerate(results['most_common_tokens'], 1):
        print(f"{i:2d}. '{token}' -> {count:,} occurrences")

    # Get coverage thresholds
    coverage_thresholds = analyzer.get_coverage_thresholds()
    print(f"\n=== Coverage Thresholds ===")
    for coverage, vocab_size in sorted(coverage_thresholds.items()):
        print(f"{coverage:.1%} coverage: {vocab_size:,} tokens")

    # Save analysis results
    output_dir = "wikipedia_token_analysis"
    print(f"\nSaving analysis results to {output_dir}/...")
    analyzer.save_analysis(output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()