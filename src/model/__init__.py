"""
Model factory.
"""

from .architectures.symbolic import SymbolicTransformer
from .architectures.vanilla import VanillaTransformer
from .architectures.tft import TFTTransformer

def get_model(model_type, config):
    """
    Factory function to create model instances.
    Returns: Model instance
    """

    models = {
        'symbolic': SymbolicTransformer,
        'vanilla': VanillaTransformer,
        'tft': TFTTransformer,
        'SymbolicTransformer': SymbolicTransformer, #alias
        'VanillaTransformer': VanillaTransformer,
        'TFTTransformer': TFTTransformer,
    }
    
    if model_type not in models:
        available = ', '.join(models.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    return models[model_type](config)

__all__ = [
    'get_model',
    'SymbolicTransformer', 
    'VanillaTransformer',
    'TFTTransformer'
]