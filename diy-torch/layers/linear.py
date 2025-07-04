import numpy as np
from core.module import Module
from core.parameter import Parameter

class Linear(Module):
    """
    A linear (fully connected) layer for neural networks.
    
    This layer applies a linear transformation to the input data: y = xW^T + b,
    where W is the weight matrix and b is the bias vector (optional). This is
    equivalent to PyTorch's nn.Linear and is one of the most fundamental building
    blocks in neural networks.
    
    The layer learns a weight matrix of shape (output_features, input_features)
    and optionally a bias vector of shape (output_features,). During the forward
    pass, it performs matrix multiplication between the input and the transposed
    weight matrix, then adds the bias if present.
    
    Args:
        input_features (int): Size of each input sample (number of input features).
        output_features (int): Size of each output sample (number of output features).
        bias (bool, optional): If set to False, the layer will not learn an additive
                              bias. Default: True.
    
    Attributes:
        weight (Parameter): The learnable weights of shape (output_features, input_features).
        bias (Parameter or None): The learnable bias of shape (output_features,).
                                 If bias=False, this will be None.
    
    Example:
        >>> # Create a linear layer that maps 784 features to 128
        >>> linear = Linear(784, 128)
        >>> input_tensor = np.random.randn(32, 784)  # batch_size=32
        >>> output = linear(input_tensor)            # shape: (32, 128)
        >>> 
        >>> # Create a linear layer without bias
        >>> linear_no_bias = Linear(10, 5, bias=False)
    """
    def __init__(self, input_features, output_features, bias=True):
        """
        Initialize the Linear layer with specified input and output dimensions.
        
        This constructor creates the weight matrix and optional bias vector with
        appropriate random initialization. The weights are initialized using
        Kaiming/He initialization (scaled by sqrt(1/input_features)) which helps
        maintain proper gradient flow in deep networks.
        
        Args:
            input_features (int): Number of input features (columns in input matrix).
            output_features (int): Number of output features (neurons in this layer).
            bias (bool): Whether to include a learnable bias term. Default: True.
        
        Example:
            >>> layer = Linear(784, 256)  # MNIST input to hidden layer
            >>> print(layer.weight.data.shape)  # (256, 784)
            >>> print(layer.bias.data.shape)    # (256,)
        """
        super().__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        self.input = None  # Store input for backward pass
        
        std = np.sqrt(1.0 / input_features)
        self.weight = Parameter(np.random.randn(output_features, input_features) * std)
        
        if bias:
            self.bias = Parameter(np.zeros(output_features))
        else:
            self.bias = None

    def forward(self, *inputs):
        """
        Perform the forward pass of the linear layer.
        
        This method applies the linear transformation y = xW^T + b to the input.
        The input is matrix-multiplied with the weight matrix, and the bias is
        added if present. This is the core computation of a fully connected layer.
        
        Args:
            *inputs: Variable number of inputs. For Linear layer, expects a single
                    input tensor of shape (batch_size, input_features) or
                    (input_features,) for single samples.
        
        Returns:
            np.ndarray: The output tensor after applying the linear transformation.
                       Shape will be (batch_size, output_features) or 
                       (output_features,) depending on input shape.
        
        Example:
            >>> layer = Linear(10, 5)
            >>> x = np.random.randn(32, 10)  # batch of 32 samples
            >>> y = layer.forward(x)         # shape: (32, 5)
            >>> 
            >>> # Single sample
            >>> x_single = np.random.randn(10)
            >>> y_single = layer.forward(x_single)  # shape: (5,)
        """
        if len(inputs) != 1:
            raise ValueError("Linear layer expects exactly one input matrix")
        
        x = inputs[0]
        self.input = x  # Store input for backward pass
        output = x @ self.weight.data.T
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.data
            
        return output

    def backward(self, grad_output):
        """
        Perform the backward pass of the linear layer.
        
        This method computes gradients with respect to the layer's parameters
        (weights and bias) and propagates gradients back to the input for the
        chain rule in backpropagation.
        
        Mathematical derivation:
        - Forward: y = x @ W^T + b
        - grad_weight = x^T @ grad_output  (accumulated)
        - grad_bias = sum(grad_output, axis=0)  (accumulated)
        - grad_input = grad_output @ W  (returned for chain rule)
        
        Args:
            grad_output (np.ndarray): Gradient of loss with respect to layer output.
                                    Shape: (batch_size, output_features) or (output_features,)
        
        Returns:
            np.ndarray: Gradient of loss with respect to layer input.
                       Shape matches self.input. Used for backprop to previous layers.
        
        Side Effects:
            - Accumulates gradients in self.weight.grad
            - Accumulates gradients in self.bias.grad (if bias exists)
        
        Example:
            >>> layer = Linear(10, 5)
            >>> x = np.random.randn(32, 10)
            >>> y = layer.forward(x)
            >>> grad_out = np.random.randn(32, 5)  # Gradient from next layer
            >>> grad_in = layer.backward(grad_out)  # Shape: (32, 10)
            >>> print(layer.weight.grad.shape)     # (5, 10)
            >>> print(layer.bias.grad.shape)       # (5,)
        """
        if self.input is None:
            raise ValueError("Must call forward() before backward()")
        
        # Handle both batch and single sample cases
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        if self.input.ndim == 1:
            input_batch = self.input.reshape(1, -1)
        else:
            input_batch = self.input
        
        # Compute gradient w.r.t weights: dL/dW = x^T @ grad_output
        # Weight shape: (output_features, input_features)
        # grad_output shape: (batch_size, output_features)  
        # input shape: (batch_size, input_features)
        grad_weight = grad_output.T @ input_batch  # Shape: (output_features, input_features)
        
        # Accumulate gradients (don't overwrite - gradients accumulate across batches)
        self.weight.grad += grad_weight
        
        # Compute gradient w.r.t bias: dL/db = sum(grad_output, axis=0)
        if self.bias is not None:
            grad_bias = grad_output.sum(axis=0)  # Shape: (output_features,)
            self.bias.grad += grad_bias
        
        # Compute gradient w.r.t input: dL/dx = grad_output @ W
        # This is passed back to the previous layer
        grad_input = grad_output @ self.weight.data  # Shape: (batch_size, input_features)
        
        # Return to original shape if single sample
        if single_sample:
            grad_input = grad_input.reshape(-1)
        
        return grad_input

    def __repr__(self):
        """
        Return a string representation of the Linear layer.
        
        This method provides a clear, informative string showing the layer's
        configuration, including input/output dimensions and whether bias is used.
        
        Returns:
            str: A string representation in the format:
                 Linear(input_features=X, output_features=Y, bias=True/False)
        
        Example:
            >>> layer = Linear(784, 256)
            >>> print(layer)  # Linear(input_features=784, output_features=256, bias=True)
        """
        return f"Linear(input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None})"