#!/usr/bin/env python3
"""
Quick script to check if dataset stories are mixed or segregated.
"""

from datasets import load_from_disk

def check_dataset_mixing():
    """Check if stories are mixed or segregated by type."""
    
    dataset = load_from_disk("./outputs/combined_data")
    print(f"Total stories: {len(dataset)}")
    
    # Sample stories from different parts of dataset
    sample_indices = [
        0, 100, 1000,                    # Beginning
        len(dataset)//4,                 # 25%
        len(dataset)//2,                 # 50% 
        3*len(dataset)//4,               # 75%
        len(dataset)-1000, len(dataset)-100, len(dataset)-1  # End
    ]
    
    print("\n🔍 STORY SAMPLES FROM DIFFERENT POSITIONS:")
    print("=" * 60)
    
    for i, idx in enumerate(sample_indices):
        if idx < len(dataset):
            story = dataset[idx]['text']
            
            # Try to identify story type by content patterns
            is_reasoning = any(phrase in story.lower() for phrase in [
                'figured out', 'realized', 'understood', 'learned that',
                'concluded', 'discovered', 'pattern', 'rule', 'because'
            ])
            
            is_simple = any(phrase in story.lower() for phrase in [
                'once upon', 'there was', 'one day', 'happily ever after',
                'the end'
            ])
            
            story_type = "REASONING" if is_reasoning else "SIMPLE" if is_simple else "MIXED"
            
            print(f"\nPosition {idx:,} ({(idx/len(dataset)*100):.1f}%): {story_type}")
            print(f"Preview: {story[:100]}{'...' if len(story) > 100 else ''}")

def suggest_shuffling():
    """Suggest how to shuffle the dataset."""
    
    print(f"\n💡 IF YOUR DATASET ISN'T MIXED:")
    print("=" * 40)
    print("You should shuffle it before training!")
    print()
    print("Option 1: Shuffle during DataLoader creation")
    print("  - Set shuffle=True in your DataLoader (already done)")
    print("  - This shuffles each epoch, which helps")
    print()
    print("Option 2: Pre-shuffle the entire dataset")
    print("  - More thorough mixing")
    print("  - See the shuffle script below")

if __name__ == "__main__":
    check_dataset_mixing()
    suggest_shuffling()