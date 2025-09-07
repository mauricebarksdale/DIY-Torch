import numpy as np
from diy_torch.nn.optim import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    SGD is one of the most fundamental optimization algorithms for neural networks.
    It updates parameters by moving them in the direction opposite to the gradient,
    scaled by the learning rate. Optionally supports momentum for smoother convergence.
    
    Parameter update rule:
    - Without momentum: param = param - lr * grad
    - With momentum: 
        velocity = momentum * velocity + grad
        param = param - lr * velocity
    
    Args:
        params (list): List of Parameter objects to optimize.
        lr (float, optional): Learning rate. Default: 0.01.
        momentum (float, optional): Momentum factor (0 <= momentum < 1). Default: 0.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 0.
    
    Example:
        >>> # Basic SGD
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> 
        >>> # SGD with momentum
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> 
        >>> # SGD with weight decay
        >>> optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    """
    
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        """
        Initialize the SGD optimizer.
        
        Args:
            params (list): List of Parameter objects to optimize.
            lr (float, optional): Learning rate. Default: 0.01.
            momentum (float, optional): Momentum factor (0 <= momentum < 1). 
                                      If 0, no momentum is applied. Default: 0.
            weight_decay (float, optional): Weight decay (L2 penalty) coefficient.
                                          If 0, no weight decay is applied. Default: 0.
        
        Raises:
            ValueError: If momentum is not in [0, 1) or weight_decay is negative.
        
        Example:
            >>> params = model.parameters()
            >>> optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)
        """
        # Validate momentum
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        
        # Validate weight decay
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        
        # Initialize base class with additional hyperparameters
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize velocity buffers for momentum (if momentum > 0)
        if self.momentum > 0:
            for param in self.params:
                param_id = id(param)
                self.state[param_id] = {
                    'velocity': np.zeros_like(param.data)
                }
    
    def step(self):
        """
        Perform a single SGD optimization step.
        
        Updates all parameters using the SGD update rule with optional momentum
        and weight decay. The update is applied in-place to parameter.data.
        
        The update steps are:
        1. Apply weight decay if specified (adds weight_decay * param to gradient)
        2. Update velocity with momentum if specified
        3. Update parameter: param = param - lr * effective_gradient
        
        Example:
            >>> # Typical training step
            >>> optimizer.zero_grad()     # Clear gradients
            >>> loss = criterion(model(x), y)  # Forward pass
            >>> loss.backward()           # Compute gradients
            >>> optimizer.step()          # Update parameters
        """
        for param in self.params:
            if param.grad is None:
                continue  # Skip parameters with no gradients
            
            grad = param.grad.copy()  # Work with a copy to avoid modifying original
            
            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            # Apply momentum if specified
            if self.momentum > 0:
                param_id = id(param)
                if param_id not in self.state:
                    # Initialize velocity buffer if not exists (shouldn't happen normally)
                    self.state[param_id] = {'velocity': np.zeros_like(param.data)}
                
                velocity = self.state[param_id]['velocity']
                
                # Update velocity: v = momentum * v + grad
                velocity *= self.momentum
                velocity += grad
                
                # Use velocity as the effective gradient
                grad = velocity
            
            # Update parameter: param = param - lr * grad
            param.data -= self.lr * grad
    
    def zero_grad(self):
        """
        Clear gradients of all parameters.
        
        This is inherited from the base Optimizer class but documented here
        for completeness in the SGD context.
        
        Example:
            >>> optimizer.zero_grad()  # Clear all parameter gradients
        """
        super().zero_grad()
    
    def __repr__(self):
        """
        Return a string representation of the SGD optimizer.
        
        Returns:
            str: String describing the SGD optimizer and its hyperparameters.
        
        Example:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> print(optimizer)  # SGD(params=10, lr=0.01, momentum=0.9, weight_decay=0)
        """
        param_count = len(self.params)
        return (f"SGD(params={param_count}, lr={self.lr}, "
                f"momentum={self.momentum}, weight_decay={self.weight_decay})")
