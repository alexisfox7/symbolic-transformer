# Inference Hooks Usage Guide

The inference hooks system allows you to observe and extract model internals during text generation. This is useful for research, interpretability, and building attention-based knowledge graphs.

## Quick Start

### 1. Run with an existing checkpoint

```bash
# Basic usage
python run_inference_with_hooks.py your_checkpoint.pt --prompt "Hello world"

# With custom parameters
python run_inference_with_hooks.py your_checkpoint.pt \
    --prompt "Once upon a time" \
    --max-tokens 50 \
    --temperature 0.8 \
    --top-k 50 \
    --save-attention attention_data.json

# Track FFN activations too
python run_inference_with_hooks.py your_checkpoint.pt \
    --prompt "Test prompt" \
    --track-activations
```

### 2. Programmatic usage

```python
from src.model import get_model
from src.config.config import TransformerConfig
from src.inference.generation import run_generation
from src.inference.hooks import create_attention_extraction_hook
from src.mytokenizers import create_tokenizer

# Load your trained model
checkpoint = torch.load('your_checkpoint.pt')
config = TransformerConfig(**checkpoint['config'])
model = get_model('vanilla', config)  # or 'symbolic'
model.load_state_dict(checkpoint['model_state_dict'])

# Create hooks
attention_hook = create_attention_extraction_hook(threshold=0.1)

# Generate with hooks
tokenizer = create_tokenizer('character')
ids, text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="Your prompt here",
    device=device,
    max_new_tokens=20,
    hooks=[attention_hook]
)

# Analyze results
print(f"Generated: {text}")
print(f"Attention records: {len(attention_hook.attention_data)}")
```

## Available Hooks

### 1. AttentionExtractionHook
Extracts attention patterns for knowledge graph construction.

```python
from src.inference.hooks import create_attention_extraction_hook

hook = create_attention_extraction_hook(
    threshold=0.1,        # Only track weights > 0.1
    store_values=False    # Store value vectors for deeper analysis
)

# After generation, access data
attention_data = hook.attention_data
token_summary = hook.get_token_attention_summary("word")
layer_edges = hook.get_edges_for_layer_head(layer=0, head=1)
```

### 2. SymbolicStreamHook
Tracks symbolic vs contextual stream activations (for symbolic transformers).

```python
from src.inference.hooks import SymbolicStreamHook

hook = SymbolicStreamHook()
# Access stream_data after generation
```

### 3. ActivationHook
Tracks intermediate FFN activations.

```python
from src.inference.hooks import ActivationHook

hook = ActivationHook(layers_to_track=[0, 1, 2])
# Access activations after generation
```

### 4. Custom Hooks
Create your own hooks by inheriting from `InferenceHook`:

```python
from src.inference.hooks import InferenceHook

class MyCustomHook(InferenceHook):
    def __init__(self):
        super().__init__("my_hook")
        self.data = []
    
    def on_attention_computed(self, layer_idx, head_idx, attention_weights, 
                            query, key, value, tokens, position, state):
        # Your custom logic here
        self.data.append({
            'layer': layer_idx,
            'head': head_idx,
            'max_attention': attention_weights.max().item()
        })
```

## Checkpoint Format

The system expects checkpoints saved in this format:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,  # TransformerConfig as dict
    'training_result': {
        'model_type': 'vanilla',  # or 'symbolic'
        # ... other training metadata
    }
}
torch.save(checkpoint, 'model.pt')
```

## Hook Events

Hooks can observe these events during generation:

- `on_generation_begin(prompt_tokens, state)` - Start of generation
- `on_generation_end(generated_tokens, state)` - End of generation  
- `on_forward_begin(input_ids, position, state)` - Before each forward pass
- `on_forward_end(logits, position, state)` - After each forward pass
- `on_attention_computed(layer_idx, head_idx, attention_weights, query, key, value, tokens, position, state)` - After attention computation
- `on_ffn_computed(layer_idx, ffn_input, ffn_output, tokens, position, state)` - After FFN computation

## Attention Analysis

The attention extraction hook provides several analysis methods:

```python
# Get attention summary for a specific token
summary = hook.get_token_attention_summary("the")
print(f"Total attention received: {summary['total_received']}")
print(f"Total attention given: {summary['total_given']}")

# Get edges for specific layer/head
edges = hook.get_edges_for_layer_head(layer=0, head=0)
for edge in edges[:5]:  # Top 5
    print(f"{edge['source_token']} -> {edge['target_token']}: {edge['weight']}")

# Access full attention records
for record in hook.attention_data:
    print(f"Layer {record['layer']}, Head {record['head']}")
    print(f"Position {record['position']}")
    print(f"Edges: {len(record['edges'])}")
```

## Testing

Run the test suite to verify everything works:

```bash
python test_hooks.py
```

Create test checkpoints for experimentation:

```bash
python create_test_checkpoint.py
```

## Examples

See these files for complete examples:
- `run_inference_with_hooks.py` - Full CLI tool
- `example_checkpoint_inference.py` - Simple programmatic example
- `test_hooks.py` - Test suite with usage examples

## Performance Notes

- Hooks add minimal overhead when disabled
- Attention extraction can use significant memory for long sequences
- Set appropriate thresholds to limit stored data
- Use `store_values=False` unless you need value vectors