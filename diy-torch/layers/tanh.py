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
        self.output = None  # Store output for backward pass

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
        self.output = np.tanh(x)  # Store output for backward pass
        
        return self.output
    
    def backward(self, grad_output):
        """
        Perform the backward pass of the Tanh activation function.
        
        The derivative of Tanh is:
        d/dx tanh(x) = 1 - tanh²(x)
        
        This is computed efficiently using the stored output from the forward pass,
        avoiding the need to recompute the expensive hyperbolic tangent operations.
        
        Args:
            grad_output (np.ndarray): Gradient of loss with respect to Tanh output.
                                    Same shape as the input/output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to Tanh input.
                       Same shape as input.
        
        Example:
            >>> tanh = Tanh()
            >>> x = np.array([[0.0, 1.0, -1.0]])
            >>> y = tanh.forward(x)  # [[0.0, 0.76, -0.76]]
            >>> grad_out = np.array([[1.0, 1.0, 1.0]])
            >>> grad_in = tanh.backward(grad_out)  # [[1.0, 0.42, 0.42]]
        """
        if self.output is None:
            raise ValueError("Must call forward() before backward()")
        
        # Tanh derivative: 1 - tanh²(x)
        # Use stored output to avoid recomputing expensive hyperbolic tangent
        grad_input = grad_output * (1 - self.output**2)
        
        return grad_input
    
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
