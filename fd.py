with open('train_vanilla_colab.py', 'w') as f:
    f.write('''#!/usr/bin/env python3
"""
Train Vanilla Transformer in Google Colab with single GPU
Simple, clean script with accelerate support
"""

#!/usr/bin/env python3
"""
Train Vanilla Transformer (GPT-2 Small Size) in Google Colab
Configured for 124M parameter model matching GPT-2 small architecture
"""

import os
import torch
import warnings
from accelerate import Accelerator
from torch.utils.data import DataLoader

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

def create_gpt2_small_config():
    """Create config matching GPT-2 small (124M parameters)"""
    config = TransformerConfig(
        # GPT-2 small architecture
        n_layer=12,        # 12 transformer blocks
        n_head=12,         # 12 attention heads
        n_embd=768,        # 768 hidden dimension

        # Training settings for Colab
        block_size=256,    # Sequence length (can increase to 512 or 1024 if memory allows)
        batch_size=4,      # Small batch size for memory efficiency
        learning_rate=6e-4,  # Standard for this size
        weight_decay=0.01,
        dropout=0.1,

        # Other settings
        num_epochs=5,      # Adjust based on your needs
        bias=True,         # GPT-2 uses bias
        use_sparsemax=False,
        use_early_exit=False,
    )
    return config

def main():
    # Parse arguments
    parser = create_base_parser("Train GPT-2 Small Size Vanilla Transformer")
    parser.add_argument('--output_dir', type=str, default='./outputs/gpt2_small')
    args = parser.parse_args()

    # Override with GPT-2 small configuration
    args.trainer_type = 'accelerate'
    args.n_layer = 12
    args.n_head = 12
    args.n_embd = 768
    args.block_size = 256  # Can increase if GPU memory allows

    # Training settings optimized for Colab GPU
    args.batch_size = 4  # Small batch for 124M model on Colab GPU
    args.num_epochs = 5
    args.learning_rate = 6e-4
    args.weight_decay = 0.01
    args.dropout = 0.1
    args.bias = True

    # Dataset settings
    args.max_samples = 100000  # Adjust based on training time
    args.val_ratio = 0.1
    args.validate_every = 500  # Validate less frequently to save time

    # Logging
    args.log_interval = 100
    args.json_log_steps = 100
    args.clip_grad_norm = 1.0

    # Optional: Enable gradient checkpointing to save memory
    # args.gradient_checkpointing = True

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.2f} GB)")

        # Adjust batch size based on GPU
        if "T4" in gpu_name:
            args.batch_size = 2  # T4 has 16GB
            print("Detected T4 GPU, using batch_size=2")
        elif "V100" in gpu_name:
            args.batch_size = 4  # V100 has 16GB but faster
            print("Detected V100 GPU, using batch_size=4")
        elif "A100" in gpu_name:
            args.batch_size = 8  # A100 has 40GB
            print("Detected A100 GPU, using batch_size=8")

    # Setup environment
    logger, device = setup_training_environment(args.output_dir, "GPT-2 Small Vanilla", args.trainer_type)

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
    print("\n" + "="*60)
    print("GPT-2 SMALL CONFIGURATION (124M Parameters)")
    print("="*60)
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
    print(f"  Max Samples:   {args.max_samples}")

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

    # Setup data loaders
    train_dataloader, val_dataloader, tokenizer = setup_data_loaders_with_combined(
        args, config, tokenizer, logger, args.trainer_type
    )

    # Create model
    logger.info("Creating GPT-2 Small Vanilla Transformer...")
    model = get_model("vanilla", config=config).to(device)

    # Count actual parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {num_params/1e6:.2f}M parameters")

    # Memory estimate
    param_memory_gb = (num_params * 4) / 1e9  # 4 bytes per parameter
    optimizer_memory_gb = param_memory_gb * 2  # Adam needs 2x parameter memory
    activation_memory_gb = (args.batch_size * config.block_size * config.n_embd * config.n_layer * 4) / 1e9
    total_memory_gb = param_memory_gb + optimizer_memory_gb + activation_memory_gb

    print(f"\nMemory Estimates:")
    print(f"  Parameters:    {param_memory_gb:.2f} GB")
    print(f"  Optimizer:     {optimizer_memory_gb:.2f} GB")
    print(f"  Activations:   {activation_memory_gb:.2f} GB")
    print(f"  Total:         {total_memory_gb:.2f} GB")

    if total_memory_gb > gpu_memory * 0.9:  # Leave 10% buffer
        print(f"⚠️  Warning: Estimated memory ({total_memory_gb:.1f}GB) may exceed GPU memory!")
        print(f"   Consider reducing batch_size or sequence length")

    # Setup optimizer with GPT-2 settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),  # GPT-2 beta values
        weight_decay=config.weight_decay,
        eps=1e-8
    )

    # Create trainer with hooks
    trainer = setup_trainer_with_hooks(
        args.trainer_type, model, train_dataloader, optimizer, device,
        config, args, val_dataloader, "GPT-2-Small-Vanilla"
    )

    # Train
    logger.info("Starting GPT-2 small training...")
    print("\n🚀 Training started! This will take a while...")
    print("   Monitor GPU usage: nvidia-smi")
    print("   Check logs: tail -f outputs/gpt2_small/training.log\n")

    training_result = trainer.train()

    # Test generation
    if not args.skip_generation:
        print("\n" + "="*60)
        print("TESTING GENERATION")
        print("="*60)
        test_generation(model, tokenizer, device, args, logger, "gpt2-small-vanilla", args.trainer_type)

    logger.info("GPT-2 small training completed!")

    # Save final model
    save_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_result': training_result
    }, save_path)
    print(f"\n✅ Model saved to: {save_path}")

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
    print("="*60 + "\n")

    # Run training
    main()
''')

print("✓ Created train_vanilla_colab.py")
!ls -la train_vanilla_colab.py