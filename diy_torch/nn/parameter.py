import numpy as np

class Parameter:
    """
    A trainable parameter for neural network modules in DIY-Torch.
    
    Parameter represents a tensor that requires gradient computation during
    backpropagation. It's similar to PyTorch's nn.Parameter and serves as
    the fundamental building block for learnable weights and biases in neural
    networks.
    
    Each Parameter contains both the actual data (weights/biases) and space
    to store gradients computed during the backward pass. This enables
    automatic gradient tracking and parameter updates during training.
    
    Attributes:
        data (np.ndarray): The actual parameter values (weights, biases, etc.)
                          stored as float32 for memory efficiency.
        grad (np.ndarray): The gradient of the loss with respect to this parameter,
                          computed during backpropagation.
    
    Example:
        >>> weight = Parameter(np.random.randn(10, 5))
        >>> bias = Parameter(np.zeros(5))
        >>> # During forward pass, use weight.data and bias.data
        >>> # During backward pass, gradients accumulate in weight.grad and bias.grad
    """
    def __init__(self, data):
        """
        Initialize a new Parameter with the given data.
        
        This constructor creates a trainable parameter by wrapping a numpy array
        with gradient tracking capabilities. The data is converted to float32
        for memory efficiency and compatibility with most neural network operations.
        
        Args:
            data (array-like): The initial values for this parameter. Can be a
                              numpy array, list, or any array-like structure.
                              Common examples include weight matrices and bias vectors.
        
        Example:
            >>> # Create weight matrix for a linear layer
            >>> weight = Parameter(np.random.randn(784, 128) * 0.01)
            >>> 
            >>> # Create bias vector
            >>> bias = Parameter(np.zeros(128))
            >>> 
            >>> # Create from list
            >>> custom_param = Parameter([[1.0, 2.0], [3.0, 4.0]])
        """
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)


    def backward(self, gradient):
        """
        Accumulate gradients for this parameter.
        
        This method adds the incoming gradient to the existing gradient accumulator.
        Gradient accumulation is essential for batch processing and for parameters
        that are used multiple times in a single forward pass.
        
        Args:
            gradient (np.ndarray): The gradient of the loss with respect to this
                                parameter, computed during backpropagation.
        
        Example:
            >>> param = Parameter(np.random.randn(10, 5))
            >>> # During backward pass
            >>> param.backward(computed_gradient)
            >>> # Gradients are now stored in param.grad
        """
        if gradient.shape != self.data.shape:
            raise ValueError(f"Gradient shape {gradient.shape} doesn't match parameter shape {self.data.shape}")
        
        self.grad += gradient.astype(np.float32)

    def zero_grad(self):
        """
        Reset the gradient of this parameter to zero.
        
        This method clears any accumulated gradients by setting the grad array
        to zeros with the same shape as the parameter data. This is essential
        before each backward pass to prevent gradient accumulation from previous
        iterations.
        
        This method is typically called automatically by Module.zero_grad(),
        which recursively zeros gradients for all parameters in a model.
        
        Example:
            >>> param = Parameter(np.random.randn(10, 5))
            >>> # After some backward pass, param.grad contains gradients
            >>> param.zero_grad()  # Clear gradients for next iteration
            >>> assert np.allclose(param.grad, 0.0)
        """
        self.grad = np.zeros_like(self.data)