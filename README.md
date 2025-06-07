# Symbolic Transformer

A pure symbolic transformer architecture where all internal representations are constrained to remain interpretable as combinations of vocabulary embeddings.

## Quick Start

### 1. Setup
```bash
# Clone/create directory
mkdir symbolic_transformer
cd symbolic_transformer

# Install dependencies
pip install -r requirements.txt

# Check setup
python check_dependencies.py
```

### 2. Demo
```bash
# Run quick demo
python demo_symbolic.py
```

### 3. Train
```bash
# Quick test (1000 samples)
python examples/train_symbolic_example.py --use_proj --use_v --batch_size 32 --max_samples 1000

# Sample training run 
python examples/train_symbolic_example.py --use_proj --use_v --batch_size 32 --max_epochs 5 --max_samples 1000000

# Or use script
./scripts/train.sh
```

## Architecture

The Symbolic Transformer enforces three key constraints:

1. **Vocabulary-Constrained FFN**: All feed-forward outputs are projected to vocabulary embedding space
2. **Channel-Wise LayerNorm**: Preserves attention head structure through symbolic normalization  
3. **ALiBi Positional Encoding**: Avoids positional embeddings that would contaminate symbolic representations

### Key Components

```python
# Symbolic constraints throughout the model
class SymbolicLayerNorm(nn.Module):
    # Preserves head channel structure
    
class VocabularyProjectionFFN(nn.Module):  
    # Projects outputs to vocabulary manifold
    
class SymbolicCausalSelfAttentionALiBi(nn.Module):
    # ALiBi + optional vocabulary constraints
```

## Configuration

### Model Presets
- **tiny**: 2L-2H-128D (for testing)
- **small**: 6L-6H-192D (quick training) 
- **medium**: 6L-6H-384D (experiments)
- **large**: 12L-12H-768D (research)

### Symbolic Options
```python
config = SymbolicConfig(
    use_symbolic_ffn=True,      # Vocabulary-constrained FFN
    use_vocab_refinement=False, # Refinement layers
    use_v=True,                 # Value constraints
    use_proj=True,              # Output constraints
)
```

## Examples

### Basic Training
```bash
python train_symbolic.py \
  --dataset "roneneldan/TinyStories" \
  --preset small \
  --max_samples 50000 \
  --batch_size 32 \
  --num_epochs 5 \
  --save_model
```

### Advanced Options
```bash
python train_symbolic.py \
  --preset medium \
  --use_vocab_refinement \
  --learning_rate 2e-4 \
  --block_size 256 \
  --output_dir "./my_model"
```

## Key Features

###  **Complete Interpretability**
Every internal representation can be interpreted as vocabulary token combinations:

```python
# Analyze symbolic content
vocab_weights = model.vocab_grounding.vocab_attention(hidden_state)
top_tokens = torch.topk(vocab_weights, k=5)
```

###  **Symbolic Reasoning**
Designed for tasks requiring explicit symbolic manipulation:
- Sequence completion (A B C → D)
- Logical inference (If A then B → ...)  
- Mathematical patterns (2+2=4, 3+3=6 → ...)

###  **ALiBi Length Extrapolation**
Can process sequences longer than training length without degradation.

## File Structure

```
symbolic_transformer/
├── README.md
├── requirements.txt  
├── config.py                 # Model configuration
├── demo_symbolic.py          # Quick demo
├── train_symbolic.py         # Training script
├── check_dependencies.py     # Setup verification
├── model/
│   ├── __init__.py
│   └── symbolic_transformer.py
├── tokenizers/
├── trainers/  
├── utils/
├── inference/
└── outputs/
```

## Research Applications

- **Mechanistic Interpretability**: Study symbolic manipulation and abductive reasoning properties 
- **Neuro-Symbolic AI**: Bridge neural and symbolic approaches  
- **Scientific Computing**: Interpretable mathematical reasoning

## Contributing

1. **Issues**: Report bugs or request features
2. **Pull Requests**: Submit improvements with tests
3. **Research**: Share experimental results and benchmarks

## Citation

```bibtex
@article{kerce_fox_symbolic_transformer_2025,
  title={Transformer Interpretability Using Purely Symbolic Internal States},
  author={Clayton Kerce and Alexis Fox},
  journal={arxiv},
  year={2025}
}
```

---

*The Symbolic Transformer: Making neural reasoning explicitly symbolic and interpretable.*
