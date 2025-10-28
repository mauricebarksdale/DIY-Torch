from . import nn
from . import optimizer

# Make common classes available at top level (optional but convenient)
from .nn import Module, Parameter, Linear, ReLU, Sigmoid, Tanh, Sequential
from .nn import Loss, MSE, CrossEntropyLoss
from .optimizer import Optimizer, SGD

__version__ = "0.1.0"

__all__ = [
    'nn',
    'optimizer',
    'Module', 'Parameter', 'Linear', 'ReLU', 'Sigmoid', 'Tanh', 'Sequential',
    'Loss', 'MSE', 'CrossEntropyLoss',
    'Optimizer', 'SGD'
]