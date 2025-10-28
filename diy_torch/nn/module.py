import numpy as np
from .parameter import Parameter

class Module:
    """
    Base class for all neural network modules in DIY-Torch.
    
    This class serves as the foundation for building neural network layers and models,
    similar to PyTorch's nn.Module. It provides the core functionality for parameter
    management, forward propagation, and gradient computation.
    
    All neural network components (layers, loss functions, models) should inherit
    from this class to gain automatic parameter tracking and gradient management.
    
    Attributes:
        _parameters (dict): Dictionary storing Parameter objects for this module.
        _modules (dict): Dictionary storing child Module objects for hierarchical composition.
    
    Example:
        >>> class MyLayer(Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = Parameter(np.random.randn(out_features, in_features))
        ...         self.bias = Parameter(np.zeros(out_features))
        ...     
        ...     def forward(self, x):
        ...         return x @ self.weight.data.T + self.bias.data
    """
    def __init__(self):
        """
        Initialize a new Module instance.
        
        Sets up internal dictionaries to track parameters and child modules.
        This is the foundation for automatic parameter discovery and gradient
        management throughout the neural network.
        """
        self._parameters = {}
        self._modules = {}
        self.training = True

    def train(self, mode=True):
        """Set module to training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set module to training mode. No gradient updates will occur."""
        return self.train(False)

    def forward(self, *inputs):
        """
        Define the forward pass computation of the module.
        
        This method must be implemented by all subclasses to define how the module
        transforms its inputs. This is where the actual computation logic lives,
        such as matrix multiplications, activations, or other transformations.
        
        Args:
            *inputs: Variable number of input tensors/arrays that the module processes.
                    The exact number and shape depend on the specific module implementation.
        
        Returns:
            The transformed output(s) after applying the module's computation.
            The return type and shape depend on the specific module implementation.
        
        Raises:
            NotImplementedError: This base implementation must be overridden by subclasses.
        
        Example:
            In a Linear layer, this would perform: output = input @ weight + bias
            In an activation layer, this would apply the activation function to inputs.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Define the backward pass computation of the module.
        
        This method should be implemented by subclasses to define how gradients
        are computed and propagated backward through the module. This is essential
        for automatic differentiation and gradient-based optimization.
        
        Args:
            grad_output: Gradient of the loss with respect to the module's output.
                        The exact type and shape depend on the specific module implementation.
        
        Returns:
            Gradient of the loss with respect to the module's input(s).
            The return type and shape depend on the specific module implementation.
        
        Raises:
            NotImplementedError: This base implementation must be overridden by subclasses
                                that need gradient computation.
        
        Note:
            Not all modules need to implement backward() - for example, loss functions
            typically only need forward(). Only implement this for modules that need
            to propagate gradients to previous layers.
        
        Example:
            In a Linear layer: computes gradients w.r.t weights, bias, and input.
            In an activation layer: computes gradients w.r.t input using chain rule.
        """
        raise NotImplementedError("Modules that need gradient computation must implement backward()")

    def __call__(self, *inputs):
        """
        Make the module callable as a function.
        
        This method enables using modules like functions by directly calling the
        instance. It simply delegates to the forward() method, providing a convenient
        interface consistent with PyTorch's design.
        
        Args:
            *inputs: Variable number of input tensors/arrays to pass through the module.
        
        Returns:
            The result of calling forward(*inputs).
        
        Example:
            >>> linear = Linear(10, 5)
            >>> output = linear(input_tensor)  # Equivalent to linear.forward(input_tensor)
        """
        return self.forward(*inputs)

    def parameters(self):
        """
        Recursively collect all parameters from this module and its child modules.
        
        This method traverses the module hierarchy to gather all trainable parameters,
        enabling automatic parameter discovery for optimizers and gradient computation.
        It returns a flat list containing parameters from this module and all nested
        child modules.
        
        Returns:
            list: A list of Parameter objects containing all trainable parameters
                 in this module and its children. The list is flattened, meaning
                 nested module structures are represented as a single flat list.
        
        Example:
            >>> model = Sequential(Linear(10, 5), Linear(5, 1))
            >>> params = model.parameters()  # Gets weights and biases from both layers
            >>> optimizer = SGD(params, lr=0.01)
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_parameter(self, name, param):
        """
        Register a parameter with the module under the given name.
        
        This method adds a Parameter object to the module's parameter dictionary,
        making it discoverable by the parameters() method and available for
        gradient computation and optimization.
        
        Args:
            name (str): The name to register the parameter under. This name is used
                       to identify and access the parameter within the module.
            param (Parameter): The Parameter object to register. Must be an instance
                              of the Parameter class containing the actual tensor data
                              and gradient information.
        
        Note:
            This method is typically called internally during module initialization.
            Most users will interact with parameters through attribute assignment,
            which automatically calls this method via __setattr__.
        
        Example:
            >>> module = Module()
            >>> weight = Parameter(np.random.randn(10, 5))
            >>> module.add_parameter('weight', weight)
        """
        self._parameters[name] = param

    def add_module(self, name, module):
        """
        Register a child module with this module under the given name.
        
        This method adds a Module instance to the module's child module dictionary,
        enabling hierarchical module composition and automatic parameter discovery.
        Child modules are included when calling parameters() and other recursive
        operations.
        
        Args:
            name (str): The name to register the child module under. This name is used
                       to identify and access the module within the parent.
            module (Module): The Module instance to register as a child. Must be an
                           instance of Module or its subclasses.
        
        Note:
            This method is typically called internally during module initialization.
            Most users will interact with child modules through attribute assignment,
            which automatically calls this method via __setattr__.
        
        Example:
            >>> parent = Module()
            >>> child = Linear(10, 5)
            >>> parent.add_module('linear_layer', child)
        """
        self._modules[name] = module

    def zero_grad(self):
        """
        Reset gradients of all parameters in this module to zero.
        
        This method recursively traverses all parameters in this module and its
        child modules, calling zero_grad() on each Parameter object. This is
        essential before each backward pass to prevent gradient accumulation
        from previous iterations.
        
        This method should be called at the beginning of each training iteration,
        typically before computing the loss and calling backward().
        
        Example:
            >>> model = Linear(10, 1)
            >>> model.zero_grad()  # Clear gradients before backward pass
            >>> loss = criterion(model(x), y)
            >>> loss.backward()
            >>> optimizer.step()
        """
        for param in self.parameters():
            param.zero_grad()

    def __setattr__(self, name, value):
        """
        Custom attribute assignment to automatically register Parameters and Modules.
        
        This method intercepts attribute assignments and automatically registers
        Parameter and Module objects with the appropriate internal dictionaries.
        This enables the convenient syntax of assigning parameters and child modules
        as attributes while maintaining proper registration for gradient computation.
        
        When a Parameter is assigned, it's automatically added to _parameters.
        When a Module is assigned, it's automatically added to _modules.
        All other attributes are handled normally.
        
        Args:
            name (str): The name of the attribute being assigned.
            value: The value being assigned to the attribute.
        
        Example:
            >>> class Linear(Module):
            ...     def __init__(self, in_features, out_features):
            ...         super().__init__()
            ...         self.weight = Parameter(np.random.randn(in_features, out_features))
            ...         # weight automatically registered via __setattr__
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)