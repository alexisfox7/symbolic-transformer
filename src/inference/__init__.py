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

from .head_logit_lens import (
    HeadLogitLensHook,
    run_head_logit_lens_analysis,
    plot_head_heatmap,
    print_head_analysis,
    analyze_head_specialization
)

# export main functions
__all__ = [
    'run_generation',
    'batch_generate',
    'LogitLensHook',
    'run_logit_lens_analysis', 
    'plot_logit_lens',
    'print_logit_lens_analysis',
    'HeadLogitLensHook',
    'run_head_logit_lens_analysis',
    'plot_head_heatmap', 
    'print_head_analysis',
    'analyze_head_specialization'
]