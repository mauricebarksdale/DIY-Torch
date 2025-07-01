import numpy as np
from core.module import Module

class Tanh(Module):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    The Tanh function applies the element-wise transformation:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    This activation function maps any real-valued input to a value between -1 and 1,
    making it zero-centered (unlike sigmoid). This property can help with gradient
    flow in deep networks and is often preferred over sigmoid for hidden layers.
    
    Properties:
    - Output range: (-1, 1)
    - Zero-centered: tanh(0) = 0
    - Monotonic: Always increasing
    - Differentiable: Smooth gradient everywhere
    - S-shaped curve: Steep in the middle, flat at extremes
    - Antisymmetric: tanh(-x) = -tanh(x)
    
    Tanh has no learnable parameters, so it's a stateless transformation.
    
    Example:
        >>> tanh = Tanh()
        >>> x = np.array([[-2.0, 0.0, 2.0]])
        >>> output = tanh(x)  # Approximately [[-0.96, 0.0, 0.96]]
    """
    
    def __init__(self):
        """
        Initialize the Tanh activation function.
        
        Tanh has no learnable parameters, so initialization is minimal.
        We only call the parent constructor to set up the basic Module structure.
        """
        super().__init__()

    def forward(self, *inputs):
        """
        Apply the Tanh activation function to the input.
        
        This method applies the element-wise transformation:
        Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        Alternatively, it can be computed as: Tanh(x) = 2 * sigmoid(2x) - 1
        
        The implementation should use numerical stability techniques to prevent
        overflow for large positive or negative values.
        
        Args:
            *inputs: Variable number of inputs. For Tanh, expects a single
                    input tensor of any shape.
        
        Returns:
            np.ndarray: The output tensor after applying Tanh activation.
                       Same shape as input, with all values in range (-1, 1).
        
        Example:
            >>> tanh = Tanh()
            >>> x = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0]])
            >>> output = tanh.forward(x)
            >>> # Result approximately: [[-1.0, -0.76, 0.0, 0.76, 1.0]]
        
        Note:
            Should implement numerically stable computation to avoid overflow
            for extreme input values.
        """
        if len(inputs) != 1:
            raise ValueError("Tanh expects exactly one input tensor")
        
        x = inputs[0]
        output = np.tanh(x)
        
        return output
    
    def __repr__(self):
        """
        Return a string representation of the Tanh layer.
        
        Returns:
            str: A simple string identifier for the Tanh activation.
        
        Example:
            >>> tanh = Tanh()
            >>> print(tanh)  # Tanh()
        """
        return "Tanh()"
