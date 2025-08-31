#!/usr/bin/env python3
"""
Create Combined Dataset: TinyStories + Wikipedia
Combines TinyStories dataset with Wikipedia articles for training vanilla models.
"""

import os
import json
import warnings
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
from itertools import islice

warnings.filterwarnings("ignore")


def load_tinystories(num_samples=50000):
    """Load TinyStories dataset."""
    print(f"Loading {num_samples} TinyStories samples...")
    
    # Load TinyStories dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    stories = []
    for story in tqdm(islice(ds, num_samples), total=num_samples, desc="Loading TinyStories"):
        text = story['text']
        if text and len(text.strip()) > 50:  # Filter very short stories
            stories.append({"text": text, "source": "tinystories"})
        
        if len(stories) >= num_samples:
            break
    
    print(f"Loaded {len(stories)} TinyStories")
    return stories


def load_wikipedia(num_samples=50000, use_full_dataset=False):
    """Load Wikipedia articles."""
    if use_full_dataset:
        print("Loading full Wikipedia dataset (this may take a while)...")
        target_message = "full dataset"
    else:
        print(f"Loading {num_samples} Wikipedia articles...")
        target_message = f"{num_samples} articles"
    
    # Load Wikipedia dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    articles = []
    processed_count = 0
    
    # Set up progress tracking
    if use_full_dataset:
        # For full dataset, we don't know the total, so use indefinite progress
        pbar = tqdm(desc="Loading Wikipedia", unit=" articles")
    else:
        pbar = tqdm(total=num_samples, desc="Loading Wikipedia")
    
    try:
        for article in ds:
            text = article['text']
            processed_count += 1
            
            # Update progress bar periodically
            if processed_count % 1000 == 0:
                if use_full_dataset:
                    pbar.set_description(f"Loading Wikipedia ({len(articles)} collected)")
                    pbar.update(1000)
                
            # Filter articles
            if (text and 
                len(text) > 200 and  # Minimum length
                len(text) < 12000 and  # Increased max length for full dataset
                not text.startswith('#REDIRECT') and  # Skip redirects
                '\n\n' in text):  # Should have some structure
                
                # Clean and chunk long articles
                if len(text) > 4000:
                    # Split very long articles into chunks
                    chunks = []
                    paragraphs = text.split('\n\n')
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk + paragraph) < 2500:  # Increased chunk size
                            current_chunk += paragraph + '\n\n'
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = paragraph + '\n\n'
                    
                    # Add remaining chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Take more chunks for full dataset
                    max_chunks = 3 if use_full_dataset else 2
                    for chunk in chunks[:max_chunks]:
                        if len(chunk) > 400:
                            articles.append({"text": chunk, "source": "wikipedia"})
                else:
                    articles.append({"text": text, "source": "wikipedia"})
            
            # Update progress for limited samples
            if not use_full_dataset:
                pbar.update(1)
            
            # Break if we've reached the target for limited sampling
            if not use_full_dataset and len(articles) >= num_samples:
                break
                
            # Safety break for full dataset (prevent infinite processing)
            if use_full_dataset and processed_count > 10000000:  # 10M articles max
                print(f"\nReached safety limit of 10M processed articles")
                break
                
    except KeyboardInterrupt:
        print(f"\nInterrupted by user after processing {processed_count} articles")
    finally:
        pbar.close()
    
    print(f"Loaded {len(articles)} Wikipedia articles/chunks from {processed_count} processed articles")
    return articles


def create_combined_dataset(tinystories_samples=50000, wikipedia_samples=50000, 
                          use_full_wikipedia=False,
                          output_path="./outputs/data/tinystories_wikipedia_combined"):
    """Create combined dataset from TinyStories and Wikipedia."""
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("="*60)
    print("CREATING COMBINED TINYSTORIES + WIKIPEDIA DATASET")
    print("="*60)
    
    # Load datasets
    stories_data = load_tinystories(tinystories_samples)
    
    if use_full_wikipedia:
        wikipedia_data = load_wikipedia(use_full_dataset=True)
        print(f"Using full Wikipedia dataset: {len(wikipedia_data)} samples")
    else:
        wikipedia_data = load_wikipedia(wikipedia_samples)
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_data = stories_data + wikipedia_data
    
    # Shuffle the combined data
    import random
    random.seed(42)
    random.shuffle(combined_data)
    
    print(f"Total combined samples: {len(combined_data)}")
    print(f"  - TinyStories: {len(stories_data)} ({len(stories_data)/len(combined_data)*100:.1f}%)")
    print(f"  - Wikipedia: {len(wikipedia_data)} ({len(wikipedia_data)/len(combined_data)*100:.1f}%)")
    
    # Create HuggingFace dataset (memory-efficient for large datasets)
    print("Creating HuggingFace dataset...")
    
    if len(combined_data) > 1000000:  # Use chunked approach for large datasets
        print(f"Large dataset detected ({len(combined_data):,} samples), using chunked processing...")
        
        # Process in chunks to avoid memory issues
        chunk_size = 200000  # 200k samples per chunk
        temp_datasets = []
        
        import gc
        for i in range(0, len(combined_data), chunk_size):
            chunk = combined_data[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(combined_data)-1)//chunk_size + 1} ({len(chunk):,} samples)")
            
            # Remove source column from chunk
            chunk_clean = [{"text": item["text"]} for item in chunk]
            temp_ds = Dataset.from_list(chunk_clean)
            temp_datasets.append(temp_ds)
            
            # Clean up memory
            del chunk, chunk_clean
            gc.collect()
        
        # Concatenate all chunks
        print("Concatenating chunks...")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(temp_datasets)
        
        # Clean up
        del temp_datasets, combined_data
        gc.collect()
        
    else:
        # Use original method for smaller datasets
        dataset = Dataset.from_list(combined_data)
        # Remove source column for training (keep only text)
        dataset = dataset.remove_columns(['source'])
    
    # Save dataset
    print(f"\nSaving combined dataset to: {output_path}")
    dataset.save_to_disk(output_path)
    
    # Save metadata
    from datetime import datetime
    metadata = {
        "description": "Combined TinyStories + Wikipedia dataset",
        "tinystories_samples": len(stories_data),
        "wikipedia_samples": len(wikipedia_data),
        "use_full_wikipedia": use_full_wikipedia,
        "total_samples": len(dataset),  # Use dataset length instead of combined_data
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(dataset.features.keys())
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Combined dataset created successfully!")
    print(f"   Path: {output_path}")
    print(f"   Samples: {len(dataset)}")
    print(f"   Features: {dataset.features}")
    
    # Show sample
    print(f"\n=== SAMPLE DATA ===")
    print(f"Sample 1 (first 200 chars): {dataset[0]['text'][:200]}...")
    print(f"Sample 2 (first 200 chars): {dataset[len(dataset)//2]['text'][:200]}...")
    
    return dataset


def main():
    """Main function to create combined dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create combined TinyStories + Wikipedia dataset")
    parser.add_argument("--tinystories_samples", type=int, default=50000,
                       help="Number of TinyStories samples to include")
    parser.add_argument("--wikipedia_samples", type=int, default=50000,
                       help="Number of Wikipedia samples to include (ignored if --full-wikipedia is used)")
    parser.add_argument("--full-wikipedia", action="store_true",
                       help="Use full Wikipedia dataset instead of limited samples")
    parser.add_argument("--output_path", type=str, 
                       default="./outputs/data/tinystories_wikipedia_combined",
                       help="Output path for combined dataset")
    
    # Add preset configurations
    parser.add_argument("--preset", type=str, choices=["small", "medium", "large", "full"],
                       help="Preset configurations: small (10k+10k), medium (50k+50k), large (500k+200k), full (500k+full Wikipedia)")
    
    args = parser.parse_args()
    
    # Apply preset configurations
    if args.preset:
        if args.preset == "small":
            args.tinystories_samples = 10000
            args.wikipedia_samples = 10000
            args.full_wikipedia = False
            args.output_path = "./outputs/data/tinystories_wikipedia_small"
        elif args.preset == "medium":
            args.tinystories_samples = 50000
            args.wikipedia_samples = 50000
            args.full_wikipedia = False
            args.output_path = "./outputs/data/tinystories_wikipedia_medium"
        elif args.preset == "large":
            args.tinystories_samples = 500000
            args.wikipedia_samples = 200000
            args.full_wikipedia = False
            args.output_path = "./outputs/data/tinystories_wikipedia_large"
        elif args.preset == "full":
            args.tinystories_samples = 500000
            args.full_wikipedia = True
            args.output_path = "./outputs/data/tinystories_wikipedia_full"
        
        print(f"Using preset '{args.preset}':")
        print(f"  TinyStories: {args.tinystories_samples:,}")
        print(f"  Wikipedia: {'Full dataset' if args.full_wikipedia else f'{args.wikipedia_samples:,}'}")
        print(f"  Output: {args.output_path}")
    
    # Create combined dataset
    dataset = create_combined_dataset(
        tinystories_samples=args.tinystories_samples,
        wikipedia_samples=args.wikipedia_samples,
        use_full_wikipedia=args.full_wikipedia,
        output_path=args.output_path
    )
    
    print(f"\n🎉 Dataset creation complete! Use with train_vanilla_colab.py:")
    print(f"   python train_vanilla_colab.py --data_source combined --max_samples {len(dataset)}")
    
    # Estimate training time and tokens
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(dataset):,}")
    
    # Rough token estimation (based on earlier analysis)
    avg_tokens_per_sample = 250  # Conservative estimate
    total_tokens = len(dataset) * avg_tokens_per_sample
    print(f"   Estimated tokens: {total_tokens:,.0f} ({total_tokens/1e9:.2f}B)")
    
    if args.preset == "full":
        print(f"\n⚠️  Full Wikipedia dataset selected - this will be very large!")
        print(f"   Estimated dataset size: Several GB")
        print(f"   Training time: Many hours/days depending on hardware")


if __name__ == "__main__":
    main()