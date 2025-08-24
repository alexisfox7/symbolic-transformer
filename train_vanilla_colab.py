#!/usr/bin/env python3
"""
Train Vanilla Transformer (GPT-2 Small Size) on Wikipedia Dataset
Modified to use Wikipedia data while keeping your existing training infrastructure
"""

import os
import torch
import warnings
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from itertools import islice
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Import your project modules
from src.utils.training_utils import (
   create_base_parser, setup_training_environment, create_config_from_args,
   setup_data_loaders_with_combined, setup_trainer_with_hooks, test_generation
)
from src.config.config import print_config, TransformerConfig
from src.mytokenizers import create_tokenizer, add_reasoning_tokens
from src.model import get_model


class WikipediaDataset(Dataset):
    """Wikipedia dataset that mimics your existing dataset format"""
    def __init__(self, texts, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        print(f"Processing {len(texts)} Wikipedia articles...")
        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize text
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Create training examples from chunks
            for i in range(0, len(tokens) - block_size + 1, block_size // 2):  # 50% overlap
                chunk = tokens[i:i + block_size]
                if len(chunk) == block_size:
                    self.examples.append({
                        'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
                        'labels': torch.tensor(chunk[1:], dtype=torch.long)
                    })
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_and_prepare_wikipedia(num_articles=50000, cache_dir="./data/wikipedia_cache"):
    """Load Wikipedia and prepare it in your existing data format"""
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wikipedia_{num_articles}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        print(f"Loading cached Wikipedia data from {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print(f"Loading {num_articles} Wikipedia articles...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    
    texts = []
    for article in tqdm(islice(ds['train'], num_articles * 2), total=num_articles):
        text = article['text']
        
        # Filter and chunk long articles
        if len(text) > 200:  # Min length
            # Split very long articles into manageable chunks
            if len(text) > 4000:
                chunks = [text[i:i+2000] for i in range(0, len(text), 1500)]
                texts.extend(chunks[:2])  # Take first 2 chunks
            else:
                texts.append(text)
        
        if len(texts) >= num_articles:
            break
    
    texts = texts[:num_articles]
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(texts, f)
    print(f"Cached {len(texts)} texts to {cache_file}")
    
    return texts


def setup_wikipedia_data_loaders(args, config, tokenizer, logger, trainer_type='accelerate'):
    """Setup data loaders for Wikipedia - replaces setup_data_loaders_with_combined"""
    
    # Load Wikipedia data
    texts = load_and_prepare_wikipedia(num_articles=args.max_samples or 50000)
    
    # Split into train/val
    split_idx = int(len(texts) * (1 - args.val_ratio))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    logger.info(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")
    
    # Create datasets
    train_dataset = WikipediaDataset(train_texts, tokenizer, config.block_size)
    val_dataset = WikipediaDataset(val_texts, tokenizer, config.block_size)
    
    # Create dataloaders with your existing settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, tokenizer


def create_gpt2_small_config():
    """Create config matching GPT-2 small (124M parameters)"""
    config = TransformerConfig(
        # GPT-2 small architecture
        n_layer=12,        # 12 transformer blocks
        n_head=12,         # 12 attention heads
        n_embd=768,        # 768 hidden dimension
        
        # Training settings for Wikipedia
        block_size=256,    # Sequence length
        batch_size=4,      # Small batch size for memory efficiency
        learning_rate=3e-4,  # Lower for Wikipedia
        weight_decay=0.01,
        dropout=0.1,
        
        # Other settings
        num_epochs=3,      # Fewer epochs with more data
        bias=True,         # GPT-2 uses bias
        use_sparsemax=False,
        use_early_exit=False,
    )
    return config


def main():
    # Parse arguments
    parser = create_base_parser("Train GPT-2 Small Size Vanilla Transformer on Wikipedia")
    parser.add_argument('--output_dir', type=str, default='./outputs/gpt2_wikipedia')
    parser.add_argument('--data_source', type=str, default='wikipedia', choices=['wikipedia', 'synthetic'])
    args = parser.parse_args()
    
    # Override with GPT-2 small configuration
    args.trainer_type = 'accelerate'
    args.n_layer = 12
    args.n_head = 12
    args.n_embd = 768
    args.block_size = 256  # Can increase if GPU memory allows
    
    # Training settings optimized for Wikipedia
    args.batch_size = 4  
    args.num_epochs = 3  # Fewer epochs for Wikipedia
    args.learning_rate = 3e-4  # Lower LR for real text
    args.weight_decay = 0.01
    args.dropout = 0.1
    args.bias = True
    
    # Dataset settings
    args.max_samples = 50000  # Number of Wikipedia articles
    args.val_ratio = 0.1
    args.validate_every = 1000  # Less frequent validation
    
    # Logging
    args.log_interval = 100
    args.json_log_steps = 100
    args.clip_grad_norm = 1.0
    
    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Adjust batch size based on GPU
        if "T4" in gpu_name:
            args.batch_size = 2  # T4 has 16GB
            args.block_size = 256
            print("Detected T4 GPU, using batch_size=2, block_size=256")
        elif "V100" in gpu_name:
            args.batch_size = 4  # V100 has 16GB but faster
            args.block_size = 512
            print("Detected V100 GPU, using batch_size=4, block_size=512")
        elif "A100" in gpu_name:
            args.batch_size = 8  # A100 has 40GB
            args.block_size = 512
            print("Detected A100 GPU, using batch_size=8, block_size=512")
    
    # Setup environment
    logger, device = setup_training_environment(args.output_dir, "GPT-2 Small Wikipedia", args.trainer_type)
    
    # Create config using GPT-2 small settings
    config = create_gpt2_small_config()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.block_size = args.block_size
    
    # Initialize tokenizer
    tokenizer = create_tokenizer(args.tokenizer_type)
    tokenizer = add_reasoning_tokens(tokenizer)
    config.update_from_tokenizer(tokenizer)
    
    # Print configuration
    print("="*60)
    print("GPT-2 SMALL CONFIGURATION - WIKIPEDIA TRAINING")
    print("="*60)
    print(f"Data Source:   Wikipedia (English)")
    print(f"Architecture:")
    print(f"  Layers:        {config.n_layer}")
    print(f"  Heads:         {config.n_head}")
    print(f"  Embedding:     {config.n_embd}")
    print(f"  Vocab Size:    {config.vocab_size}")
    print(f"  Sequence Len:  {config.block_size}")
    print(f"Training:")
    print(f"  Batch Size:    {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Max Articles:  {args.max_samples}")
    
    # Calculate approximate parameter count
    approx_params = (
        12 * config.n_embd * config.vocab_size +  # Token embeddings + output
        config.n_layer * (
            4 * config.n_embd * config.n_embd +   # QKV + proj in attention
            8 * config.n_embd * config.n_embd      # FFN (4x hidden)
        )
    )
    print(f"Estimated Parameters: {approx_params/1e6:.1f}M")
    print("="*60)
    
    # Setup data loaders - Use Wikipedia instead of synthetic data
    if args.data_source == 'wikipedia':
        train_dataloader, val_dataloader, tokenizer = setup_wikipedia_data_loaders(
            args, config, tokenizer, logger, args.trainer_type
        )
    else:
        # Fall back to your original synthetic data
        train_dataloader, val_dataloader, tokenizer = setup_data_loaders_with_combined(
            args, config, tokenizer, logger, args.trainer_type
        )
    
    # Create model - using your existing model
    logger.info("Creating GPT-2 Small Vanilla Transformer...")
    model = get_model("vanilla", config=config).to(device)
    
    # Count actual parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {num_params/1e6:.2f}M parameters")
    
    # Memory estimate
    param_memory_gb = (num_params * 4) / 1e9
    optimizer_memory_gb = param_memory_gb * 2
    activation_memory_gb = (args.batch_size * config.block_size * config.n_embd * config.n_layer * 4) / 1e9
    total_memory_gb = param_memory_gb + optimizer_memory_gb + activation_memory_gb
    
    print(f"Memory Estimates:")
    print(f"  Parameters:    {param_memory_gb:.2f} GB")
    print(f"  Optimizer:     {optimizer_memory_gb:.2f} GB")
    print(f"  Activations:   {activation_memory_gb:.2f} GB")
    print(f"  Total:         {total_memory_gb:.2f} GB")
    
    # Setup optimizer with GPT-2 settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),  # GPT-2 beta values
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    # Create trainer with hooks - using your existing trainer
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "GPT-2-Wikipedia-Vanilla"
    )
    
    # Train
    logger.info("Starting Wikipedia training...")
    print("🚀 Training on Wikipedia started!")
    print(f"   Training examples: {len(train_dataloader.dataset)}")
    print(f"   Validation examples: {len(val_dataloader.dataset)}")
    print(f"   Batches per epoch: {len(train_dataloader)}")
    print("   Monitor: tail -f outputs/gpt2_wikipedia/training.log")
    
    training_result = trainer.train()
    
    # Test generation with Wikipedia-appropriate prompts
    if not args.skip_generation:
        print("\nTESTING GENERATION - WIKIPEDIA STYLE")
        print("="*60)
        
        # Wikipedia-style test prompts
        test_prompts = [
            "The United States is",
            "In computer science,",
            "The human brain",
            "Climate change refers to",
            "The history of"
        ]
        
        model.eval()
        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                # Generate using your model's generate method
                output = model.generate(
                    input_ids,
                    max_length=50,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated: {generated[:200]}...")  # Show first 200 chars
    
    logger.info("Wikipedia training completed!")
    
    # Save final model
    save_path = os.path.join(args.output_dir, "wikipedia_vanilla_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_result': training_result
    }, save_path)
    print(f"✅ Model saved to: {save_path}")
    
    return training_result


if __name__ == "__main__":
    # Check CUDA availability
    print("="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {memory_gb:.2f} GB")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")
    
    # Install datasets if needed
    print("\nChecking dependencies...")
    os.system("pip install -q datasets")
    
    # Run training
    main()