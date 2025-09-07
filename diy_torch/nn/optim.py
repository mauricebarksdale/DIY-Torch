import numpy as np

class Optimizer:
    """
    Base class for all optimizers in DIY-Torch.
    
    This class provides the foundation for implementing gradient-based optimization
    algorithms such as SGD, Adam, RMSprop, etc. Optimizers are responsible for
    updating model parameters based on computed gradients to minimize the loss
    function.
    
    The optimizer maintains a list of parameters to optimize and implements
    the parameter update logic specific to each optimization algorithm.
    
    Attributes:
        params (list): List of Parameter objects to optimize.
        lr (float): Learning rate for parameter updates.
        defaults (dict): Default hyperparameter values for the optimizer.
    
    Example:
        >>> # Create model and get its parameters
        >>> model = Sequential(Linear(10, 5), Linear(5, 1))
        >>> params = model.parameters()
        >>> 
        >>> # Create optimizer
        >>> optimizer = SGD(params, lr=0.01)
        >>> 
        >>> # Training loop
        >>> for epoch in range(num_epochs):
        ...     model.zero_grad()           # Clear gradients
        ...     loss = criterion(model(x), y)  # Forward pass
        ...     loss.backward()             # Backward pass
        ...     optimizer.step()            # Update parameters
    """
    
    def __init__(self, params, lr=0.01, **kwargs):
        """
        Initialize the base optimizer.
        
        Args:
            params (list): List of Parameter objects to optimize. Typically
                          obtained from model.parameters().
            lr (float, optional): Learning rate for parameter updates. Default: 0.01.
            **kwargs: Additional hyperparameters specific to the optimizer.
        
        Raises:
            ValueError: If params is empty or lr is not positive.
        
        Example:
            >>> params = model.parameters()
            >>> optimizer = SGD(params, lr=0.001, momentum=0.9)
        """
        if not params:
            raise ValueError("Optimizer requires at least one parameter to optimize")
        
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        
        self.params = list(params)
        self.lr = lr
        self.defaults = {'lr': lr}
        self.defaults.update(kwargs)
        
        # State dictionary for storing optimizer-specific state per parameter
        self.state = {}
    
    def step(self):
        """
        Perform a single optimization step (parameter update).
        
        This method must be implemented by all subclasses to define the specific
        parameter update rule for the optimization algorithm. It typically:
        
        1. Iterates through all parameters
        2. Computes parameter updates based on gradients and optimizer state
        3. Applies the updates to parameter.data
        4. Updates any internal optimizer state
        
        Raises:
            NotImplementedError: This base implementation must be overridden by subclasses.
        
        Example:
            >>> optimizer.step()  # Updates all parameters based on their gradients
        """
        raise NotImplementedError("Subclasses must implement step() method")
    
    def zero_grad(self):
        """
        Clear gradients of all parameters managed by this optimizer.
        
        This method sets the .grad attribute of all parameters to zero,
        preparing them for the next backward pass. This is essential to
        prevent gradient accumulation across training iterations.
        
        Note:
            This is a convenience method. You can also call model.zero_grad()
            directly, which may be more efficient for complex models.
        
        Example:
            >>> optimizer.zero_grad()  # Clear all parameter gradients
            >>> # Equivalent to: model.zero_grad()
        """
        for param in self.params:
            if param.grad is not None:
                param.zero_grad()
    
    def add_param_group(self, param_group):
        """
        Add a new parameter group to the optimizer.
        
        This allows different groups of parameters to have different
        hyperparameters (e.g., different learning rates for different layers).
        
        Args:
            param_group (dict): Dictionary containing 'params' key with list
                              of parameters, and optionally other hyperparameters.
        
        Example:
            >>> # Different learning rates for different layers
            >>> backbone_params = backbone.parameters()
            >>> head_params = head.parameters()
            >>> 
            >>> optimizer = SGD(backbone_params, lr=0.001)
            >>> optimizer.add_param_group({'params': head_params, 'lr': 0.01})
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dictionary")
        
        if 'params' not in param_group:
            raise KeyError("param_group must contain 'params' key")
        
        # Add default values for missing hyperparameters
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        
        # Add parameters to the main list
        self.params.extend(param_group['params'])
    
    def get_state(self):
        """
        Get the current state of the optimizer.
        
        Returns a dictionary containing all optimizer state information,
        including hyperparameters and any internal state variables.
        This is useful for saving and loading optimizer state.
        
        Returns:
            dict: Dictionary containing optimizer state information.
        
        Example:
            >>> state = optimizer.get_state()
            >>> # Save state to file or use for checkpointing
        """
        return {
            'lr': self.lr,
            'defaults': self.defaults,
            'state': self.state
        }
    
    def load_state(self, state_dict):
        """
        Load optimizer state from a dictionary.
        
        Restores the optimizer to a previous state, useful for resuming
        training from checkpoints.
        
        Args:
            state_dict (dict): Dictionary containing optimizer state,
                             typically from get_state().
        
        Example:
            >>> state = optimizer.get_state()  # Save state
            >>> # ... later ...
            >>> optimizer.load_state(state)   # Restore state
        """
        self.lr = state_dict.get('lr', self.lr)
        self.defaults = state_dict.get('defaults', self.defaults)
        self.state = state_dict.get('state', {})
    
    def __repr__(self):
        """
        Return a string representation of the optimizer.
        
        Returns:
            str: String describing the optimizer and its hyperparameters.
        """
        param_count = len(self.params)
        return f"{self.__class__.__name__}(params={param_count}, lr={self.lr})"
