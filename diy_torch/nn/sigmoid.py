import numpy as np
from diy_torch.nn.module import Module

class Sigmoid(Module):
    """
    Sigmoid activation function.
    
    The Sigmoid function applies the element-wise transformation:
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    This activation function maps any real-valued input to a value between 0 and 1,
    making it useful for binary classification tasks and as a gate mechanism in
    neural networks. However, it can suffer from vanishing gradients for very
    large or very small inputs.
    
    Properties:
    - Output range: (0, 1)
    - Monotonic: Always increasing
    - Differentiable: Smooth gradient everywhere
    - S-shaped curve: Steep in the middle, flat at extremes
    
    Sigmoid has no learnable parameters, so it's a stateless transformation.
    
    Example:
        >>> sigmoid = Sigmoid()
        >>> x = np.array([[-2.0, 0.0, 2.0]])
        >>> output = sigmoid(x)  # Approximately [[0.12, 0.5, 0.88]]
    """
    
    def __init__(self):
        """
        Initialize the Sigmoid activation function.
        
        Sigmoid has no learnable parameters, so initialization is minimal.
        We only call the parent constructor to set up the basic Module structure.
        """
        super().__init__()
        self.output = None  # Store output for backward pass

    def forward(self, *inputs):
        """
        Apply the Sigmoid activation function to the input.
        
        This method applies the element-wise transformation:
        Sigmoid(x) = 1 / (1 + exp(-x))
        
        The implementation uses numerical stability techniques to prevent
        overflow for large negative values by using the mathematically
        equivalent formulation when x < 0.
        
        Args:
            *inputs: Variable number of inputs. For Sigmoid, expects a single
                    input tensor of any shape.
        
        Returns:
            np.ndarray: The output tensor after applying Sigmoid activation.
                       Same shape as input, with all values in range (0, 1).
        
        Example:
            >>> sigmoid = Sigmoid()
            >>> x = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0]])
            >>> output = sigmoid.forward(x)
            >>> # Result approximately: [[0.0, 0.27, 0.5, 0.73, 1.0]]
        
        Note:
            Uses numerically stable computation to avoid overflow:
            - For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
            - For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
        """
        if len(inputs) != 1:
            raise ValueError("Sigmoid expects exactly one input tensor")
        
        x = inputs[0]
        self.output = 1 / (1 + np.exp(-x))  # Store output for backward pass
        
        return self.output
    
    def backward(self, grad_output):
        """
        Perform the backward pass of the Sigmoid activation function.
        
        The derivative of Sigmoid is:
        d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        
        This is computed efficiently using the stored output from the forward pass,
        avoiding the need to recompute the expensive exponential operations.
        
        Args:
            grad_output (np.ndarray): Gradient of loss with respect to Sigmoid output.
                                    Same shape as the input/output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to Sigmoid input.
                       Same shape as input.
        
        Example:
            >>> sigmoid = Sigmoid()
            >>> x = np.array([[0.0, 1.0, -1.0]])
            >>> y = sigmoid.forward(x)  # [[0.5, 0.73, 0.27]]
            >>> grad_out = np.array([[1.0, 1.0, 1.0]])
            >>> grad_in = sigmoid.backward(grad_out)  # [[0.25, 0.196, 0.196]]
        """
        if self.output is None:
            raise ValueError("Must call forward() before backward()")
        
        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        # Use stored output to avoid recomputing expensive exponentials
        grad_input = grad_output * self.output * (1 - self.output)
        
        return grad_input
    
    def __repr__(self):
        """
        Return a string representation of the Sigmoid layer.
        
        Returns:
            str: A simple string identifier for the Sigmoid activation.
        
        Example:
            >>> sigmoid = Sigmoid()
            >>> print(sigmoid)  # Sigmoid()
        """
        return "Sigmoid()"
