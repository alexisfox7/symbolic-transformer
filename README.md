# Symbolic Transformers

## Directory Structure

```
symbolic_transformer/
├── src/                        # SOURCE CODE 
│   ├── config/                 # Model config management
│   ├── inference/              # Generation & sampling utilities
│   ├── model/                  # Model architectures & components
│   │   ├── architectures/      # Different transformer variants (vanilla, symbolic, TFT)
│   │   └── components/         # Model building blocks (attention, FFN, normalization)
│   ├── mytokenizers/           # Custom tokenizer 
│   ├── trainers/               # Trainers (i.e. accelerate)
│   └── utils/                  # Training & dataloader utility functions
├── exp/                        # EXPERIMENT FILES
│   ├── examples/               # Training scripts
│   └── scripts/                # Shell scripts for training runs
├── outputs/                    # OUTPUTS
│   └── inference/              # Inference results
├── requirements.txt            # Requirements
├── setup.py                    # Package installation configuration
└── run_inference_with_hooks.py # Test out hooked inference + visualize attention matrices
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0

## Usage

Run training scripts:

```bash
bash exp/scripts/cascade_train.sh
bash exp/scripts/vanilla_train.sh
bash exp/scripts/tft_train.sh
bash exp/scripts/sym_train.sh
```

Run inference:

```bash
python run_inference_with_hooks.py
```