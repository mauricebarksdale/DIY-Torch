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
        self.input = None  # Store input for backward pass

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
        self.input = x  # Store input for backward pass
        output = np.maximum(0.0, x)
        
        return output
    
    def backward(self, grad_output):
        """
        Perform the backward pass of the ReLU activation function.
        
        The derivative of ReLU is:
        - 1 if input > 0
        - 0 if input <= 0
        
        This creates a binary mask that passes gradients through for positive
        inputs and blocks gradients for negative inputs.
        
        Args:
            grad_output (np.ndarray): Gradient of loss with respect to ReLU output.
                                    Same shape as the input/output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to ReLU input.
                       Same shape as input. Elements are either grad_output (if input > 0)
                       or 0 (if input <= 0).
        
        Example:
            >>> relu = ReLU()
            >>> x = np.array([[-1.0, 0.0, 1.0, 2.0]])
            >>> y = relu.forward(x)  # [[0.0, 0.0, 1.0, 2.0]]
            >>> grad_out = np.array([[1.0, 1.0, 1.0, 1.0]])
            >>> grad_in = relu.backward(grad_out)  # [[0.0, 0.0, 1.0, 1.0]]
        """
        if self.input is None:
            raise ValueError("Must call forward() before backward()")
        
        # ReLU derivative: 1 if input > 0, else 0
        # This creates a mask that passes gradients through for positive inputs
        mask = (self.input > 0).astype(np.float32)
        
        # Element-wise multiplication: pass gradient through where input was positive
        grad_input = grad_output * mask
        
        return grad_input
    
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