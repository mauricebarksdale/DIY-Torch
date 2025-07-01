import numpy as np
from .parameter import Parameter

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def parameters(self):
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_parameter(self, name, param):
        self._parameters[name] = param

    def add_module(self, name, module):
        self._modules[name] = module

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)