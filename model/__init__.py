# ./model/__init__.py
"""
Model factory for cleanGPT models.
"""

from .model_symbolic_transformer import SymbolicTransformerModel
from .model_vanilla_transformer import VanillaTransformerModel
from .model_symbolic_reconstruct import SymbolicTransformerModelWithReconstruction

def get_model(model_type, config=None, **kwargs):
    """
    Factory function to create model instances.
    """
    model_registry = {
        'Symbolic': SymbolicTransformerModel,
        'SymbolicTransformer': SymbolicTransformerModel,
        'SymbolicReconstruct': SymbolicTransformerModelWithReconstruction,  # NEW
        'SymbolicRecon': SymbolicTransformerModelWithReconstruction,        # NEW (alias)
        'Vanilla': VanillaTransformerModel,
        'VanillaTransformer': VanillaTransformerModel,
        'Standard': VanillaTransformerModel,
    }
    
    if model_type not in model_registry:
        available_types = ', '.join(model_registry.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available types: {available_types}")
    
    model_class = model_registry[model_type]
    
    if config is None:
        raise ValueError("Config must be provided")
    
    return model_class(config, **kwargs)

__all__ = ['get_model', 'SymbolicTransformerModel', 'SymbolicTransformerModelWithReconstruction',  'VanillaTransformerModel']