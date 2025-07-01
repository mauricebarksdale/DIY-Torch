from collections import OrderedDict
from core.module import Module

class Sequential(Module):
    """
    A sequential container for neural network modules.
    
    Sequential allows you to chain multiple modules together in a linear fashion,
    where the output of one module becomes the input to the next. This is similar
    to PyTorch's nn.Sequential and is useful for building simple feedforward
    networks or chains of operations.
    
    The modules are executed in the order they are added to the Sequential container.
    This makes it easy to build common architectures like multi-layer perceptrons
    or convolutional neural networks with clear, readable code.
    
    Args:
        *layers: Variable number of Module instances to chain together in sequence.
    
    Example:
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Linear(256, 128),
        ...     ReLU(),
        ...     Linear(128, 10)
        ... )
        >>> output = model(input_tensor)
    """
    def __init__(self, *layers):
        """
        Initialize a Sequential container with the given layers.
        
        This constructor takes a variable number of Module instances and chains
        them together in the order they are provided. Each module is registered
        as a child module with a string index as its name, enabling automatic
        parameter discovery and proper gradient flow.
        
        Args:
            *layers: Variable number of Module instances to chain together.
                    Each layer should be a subclass of Module (e.g., Linear,
                    ReLU, Conv2d, etc.).
        
        Example:
            >>> seq = Sequential(
            ...     Linear(10, 5),
            ...     ReLU(),
            ...     Linear(5, 1)
            ... )
        """
        super().__init__()
        self._layers = OrderedDict()

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def forward(self, *inputs):
        """
        Execute the forward pass through all modules in sequence.
        
        This method passes the input through each module in the order they were
        added to the Sequential container. The output of each module becomes the
        input to the next module, creating a chain of transformations.
        
        Args:
            *inputs: Variable number of input tensors/arrays. For Sequential,
                    typically expects a single input tensor, but the signature
                    matches the base Module class for consistency.
        
        Returns:
            The output after passing through all layers in sequence. The shape
            and type depend on the final layer's output.
        
        Example:
            >>> seq = Sequential(Linear(10, 5), ReLU(), Linear(5, 1))
            >>> input_tensor = np.random.randn(32, 10)  # batch_size=32, features=10
            >>> output = seq.forward(input_tensor)      # shape: (32, 1)
        """
        x = inputs
        
        for layer in self._modules.values():
            x = layer(x)
            
        return x