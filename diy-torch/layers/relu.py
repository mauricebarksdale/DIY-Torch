import numpy as np
from core.module import Module

class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU applies the element-wise function ReLU(x) = max(0, x) to the input.
    This is one of the most commonly used activation functions in neural networks
    because it's simple, computationally efficient, and helps mitigate the
    vanishing gradient problem.
    
    The ReLU function:
    - Outputs the input directly if it's positive
    - Outputs zero if the input is negative or zero
    
    ReLU has no learnable parameters, so it's a stateless transformation.
    
    Example:
        >>> relu = ReLU()
        >>> x = np.array([[-1.0, 0.0, 1.0, 2.0]])
        >>> output = relu(x)  # Result: [[0.0, 0.0, 1.0, 2.0]]
    """
    def __init__(self):
        """
        Initialize the ReLU activation function.
        
        ReLU has no learnable parameters, so initialization is minimal.
        We only call the parent constructor to set up the basic Module structure.
        """
        super().__init__()

    def forward(self, *inputs):
        """
        Apply the ReLU activation function to the input.
        
        This method applies the element-wise transformation ReLU(x) = max(0, x)
        to the input tensor. The operation is performed in-place mathematically
        but returns a new array.
        
        Args:
            *inputs: Variable number of inputs. For ReLU, expects a single
                    input tensor of any shape.
        
        Returns:
            np.ndarray: The output tensor after applying ReLU activation.
                       Same shape as input, with all negative values set to 0.
        
        Example:
            >>> relu = ReLU()
            >>> x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
            >>> output = relu.forward(x)  # [[0.0, 0.0, 0.0, 1.0, 2.0]]
        """
        if len(inputs) != 1:
            raise ValueError("ReLU expects exactly one input tensor")
        
        x = inputs[0]
        output = np.maximum(0.0, x)
        
        return output
    
    def __repr__(self):
        """
        Return a string representation of the ReLU layer.
        
        Returns:
            str: A simple string identifier for the ReLU activation.
        
        Example:
            >>> relu = ReLU()
            >>> print(relu)  # ReLU()
        """
        return "ReLU()" 