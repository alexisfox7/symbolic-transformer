#!/usr/bin/env python3
"""
Test script to verify combined dataset loading and show statistics.
"""

import os
import sys
from datasets import load_from_disk
from src.mytokenizers import create_tokenizer

def test_dataset_loading():
    """Test loading the combined dataset and show statistics."""
    
    dataset_path = "./outputs/combined_data"
    
    print("🔍 TESTING DATASET LOADING")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                print(f"  📁 {item}")
        return False
    
    try:
        # Load dataset
        print(f"📂 Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total stories: {len(dataset):,}")
        
        # Show dataset info
        print(f"\n📋 Dataset Info:")
        print(f"  Features: {list(dataset.features.keys())}")
        print(f"  Dataset type: {type(dataset)}")
        
        # Sample stories
        print(f"\n📚 Sample Stories:")
        print("-" * 30)
        
        for i in range(min(3, len(dataset))):
            story = dataset[i]['text']
            print(f"\nStory {i+1}:")
            print(f"Length: {len(story)} chars")
            print(f"Preview: {story[:150]}{'...' if len(story) > 150 else ''}")
        
        # Story length statistics
        print(f"\n📏 Story Length Statistics:")
        print("-" * 30)
        
        lengths = [len(story['text']) for story in dataset.select(range(min(1000, len(dataset))))]
        lengths.sort()
        
        print(f"  Sample size: {len(lengths)} stories")
        print(f"  Min length: {min(lengths)} chars")
        print(f"  Max length: {max(lengths)} chars")
        print(f"  Avg length: {sum(lengths)//len(lengths)} chars")
        print(f"  Median length: {lengths[len(lengths)//2]} chars")
        
        # Check for empty/short stories
        short_stories = sum(1 for length in lengths if length < 20)
        print(f"  Very short (<20 chars): {short_stories} stories")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_creation():
    """Test creating a DataLoader with the dataset."""
    
    print(f"\n🔧 TESTING DATALOADER CREATION")
    print("=" * 50)
    
    try:
        # Load dataset
        dataset = load_from_disk("./outputs/combined_data")
        
        # Create tokenizer
        print("🔤 Creating tokenizer...")
        tokenizer = create_tokenizer('gpt2')
        
        # Test the collate function
        from src.utils.data_utils import simple_collate_fn
        
        # Take a small batch
        batch_size = 4
        sample_batch = [dataset[i] for i in range(batch_size)]
        
        print(f"📦 Testing batch collation (size: {batch_size})...")
        
        # Test collate function
        collated = simple_collate_fn(sample_batch, tokenizer, max_length=128)
        
        print(f"✅ Collation successful!")
        print(f"  Input IDs shape: {collated['input_ids'].shape}")
        print(f"  Targets shape: {collated['targets'].shape}")
        
        # Show tokenized example
        print(f"\n🔍 Tokenization Example:")
        original_text = sample_batch[0]['text']
        tokenized = collated['input_ids'][0]
        decoded = tokenizer.decode(tokenized, skip_special_tokens=True)
        
        print(f"  Original: {original_text[:100]}...")
        print(f"  Tokens: {len(tokenized)} tokens")
        print(f"  Decoded: {decoded[:100]}...")
        
        # Test DataLoader creation
        from torch.utils.data import DataLoader
        
        def collate_wrapper(batch):
            return simple_collate_fn(batch, tokenizer, 128)
        
        dataloader = DataLoader(
            dataset.select(range(min(100, len(dataset)))),  # Small subset for testing
            batch_size=8,
            shuffle=True,
            collate_fn=collate_wrapper,
            drop_last=True
        )
        
        print(f"\n📊 DataLoader created successfully!")
        print(f"  Batches: {len(dataloader)}")
        print(f"  Batch size: {dataloader.batch_size}")
        
        # Test one batch
        print(f"\n🔄 Testing one batch...")
        batch = next(iter(dataloader))
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  Batch targets shape: {batch['targets'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_existing_data_utils():
    """Test using the actual data loading functions from your codebase."""
    
    print(f"\n⚙️ TESTING WITH EXISTING DATA UTILS")
    print("=" * 50)
    
    try:
        # Test if we can import the function
        from src.utils.data_utils import load_combined_tinystories
        
        tokenizer = create_tokenizer('gpt2')
        
        print("🔄 Testing load_combined_tinystories function...")
        
        dataloader, tokenizer = load_combined_tinystories(
            dataset_path="./outputs/combined_data",
            tokenizer=tokenizer,
            max_samples=1000,  # Small subset for testing
            max_seq_length=128,
            batch_size=8,
            shuffle=False
        )
        
        print(f"✅ Function works!")
        print(f"  DataLoader batches: {len(dataloader)}")
        print(f"  Batch size: {dataloader.batch_size}")
        
        # Test one batch
        batch = next(iter(dataloader))
        print(f"  Sample batch shape: {batch['input_ids'].shape}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Could not import load_combined_tinystories: {e}")
        print("  This is expected if you haven't added it to data_utils.py yet")
        return False
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🧪 DATASET LOADING TEST SUITE")
    print("=" * 60)
    
    # Test 1: Basic dataset loading
    success1 = test_dataset_loading()
    
    # Test 2: DataLoader creation  
    success2 = test_dataloader_creation() if success1 else False
    
    # Test 3: Integration with existing utils
    success3 = test_with_existing_data_utils() if success1 else False
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 30)
    print(f"  Dataset Loading:     {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  DataLoader Creation: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"  Utils Integration:   {'✅ PASS' if success3 else '⚠️ SKIP' if not success1 else '❌ FAIL'}")
    
    if success1 and success2:
        print(f"\n🎉 Ready to train! Your dataset is working correctly.")
    else:
        print(f"\n🔧 Some issues found. Check the error messages above.")

if __name__ == "__main__":
    main()