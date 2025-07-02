class Loss:
    """
    Base class for all loss functions in DIY-Torch.
    
    This class serves as the foundation for implementing various loss functions
    used in neural network training. Loss functions measure the difference between
    predicted outputs and target values, providing the objective to minimize
    during training.
    
    All loss functions should inherit from this class and implement both forward
    and backward methods to enable automatic differentiation and gradient-based
    optimization.
    
    The loss function typically:
    1. Computes the loss value in the forward pass
    2. Computes gradients with respect to inputs in the backward pass
    
    Example:
        >>> class MeanSquaredError(Loss):
        ...     def forward(self, prediction, target):
        ...         return np.mean((prediction - target) ** 2)
        ...     
        ...     def backward(self):
        ...         # Compute and return gradients
        ...         pass
    """
    def forward(self, prediction, target):
        """
        Compute the loss value between predictions and targets.
        
        This method must be implemented by all subclasses to define the specific
        loss computation. The forward pass typically involves comparing the
        predicted values with the ground truth targets and returning a scalar
        loss value.
        
        Args:
            prediction (np.ndarray): The predicted output from the model.
                                   Shape depends on the specific task and loss function.
            target (np.ndarray): The ground truth target values.
                               Shape should be compatible with predictions.
        
        Returns:
            float or np.ndarray: The computed loss value. Often a scalar for
                               batch-averaged losses, but can be an array for
                               element-wise losses.
        
        Raises:
            NotImplementedError: This base implementation must be overridden by subclasses.
        
        Example:
            >>> loss_fn = MeanSquaredError()
            >>> predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> targets = np.array([[1.5, 2.5], [2.5, 3.5]])
            >>> loss_value = loss_fn.forward(predictions, targets)
        """
        raise NotImplementedError

    def backward(self):
        """
        Compute gradients of the loss with respect to the inputs.
        
        This method must be implemented by all subclasses to define how gradients
        are computed and propagated backward through the network. The backward pass
        is essential for gradient-based optimization algorithms like SGD, Adam, etc.
        
        The implementation should:
        1. Compute gradients with respect to predictions (and possibly targets)
        2. Store or return these gradients for use by the optimization process
        3. Handle the chain rule for backpropagation through the computational graph
        
        Returns:
            np.ndarray or tuple: The computed gradients. The exact format depends
                               on the specific loss function implementation.
                               Commonly returns gradients with respect to predictions.
        
        Raises:
            NotImplementedError: This base implementation must be overridden by subclasses.
        
        Note:
            This method is typically called after forward() and assumes that
            necessary intermediate values have been stored during the forward pass
            for gradient computation.
        
        Example:
            >>> loss_fn = MeanSquaredError()
            >>> loss_value = loss_fn.forward(predictions, targets)
            >>> gradients = loss_fn.backward()  # Compute gradients for backprop
        """
        raise NotImplementedError