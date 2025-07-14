#./inference/__init__.py
"""
Inference module
Provides utilities for text generation and model inference
"""

from .generation import (
    run_generation,
    batch_generate
)

from .logit_lens import (
    LogitLensHook,
    run_logit_lens_analysis,
    plot_logit_lens,
    print_logit_lens_analysis
)

# export main functions
__all__ = [
    'run_generation',
    'batch_generate',
    'LogitLensHook',
    'run_logit_lens_analysis', 
    'plot_logit_lens',
    'print_logit_lens_analysis'
]