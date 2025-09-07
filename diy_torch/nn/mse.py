import numpy as np
from diy_torch.nn.loss import Loss

class MSE(Loss):
    """
    Mean Squared Error (MSE) loss function.
    
    MSE computes the average of the squared differences between predictions and targets:
    MSE = (1/n) * Σ(target - prediction)²
    
    This loss is commonly used for regression tasks where the goal is to predict
    continuous values. It penalizes larger errors more heavily due to the squaring.
    
    Example:
        >>> mse_loss = MSE()
        >>> predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> targets = np.array([[1.5, 2.5], [2.5, 3.5]])
        >>> loss = mse_loss.forward(predictions, targets)  # Returns scalar loss
        >>> grads = mse_loss.backward()  # Returns gradients w.r.t predictions
    """
    def __init__(self):
        """
        Initialize the MSE loss function.
        
        Sets up storage for predictions and targets needed for backward pass.
        """
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        """
        Compute the MSE loss between predictions and targets.
        
        Args:
            prediction (np.ndarray): Model predictions of any shape.
            target (np.ndarray): Ground truth targets with same shape as predictions.
        
        Returns:
            float: The mean squared error loss value.
        
        Raises:
            ValueError: If prediction or target is None, or if shapes don't match.
        """
        if prediction is None or target is None:
            raise ValueError("Prediction and target cannot be None")
        
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}")
        
        self.prediction = prediction
        self.target = target

        return np.mean((target - prediction) ** 2)
    
    def backward(self):
        """
        Compute gradients of MSE loss with respect to predictions.
        
        The gradient of MSE loss with respect to predictions is:
        ∂L/∂prediction = 2 * (prediction - target) / n
        
        Where n is the total number of elements (for mean reduction).
        
        Returns:
            np.ndarray: Gradients with respect to predictions, same shape as predictions.
        
        Raises:
            ValueError: If forward() hasn't been called yet.
        
        Example:
            The gradient tells us how much each prediction should change to reduce the loss.
            Positive gradient means prediction is too high, negative means too low.
        """
        if self.prediction is None or self.target is None:
            raise ValueError("Must call forward() before backward()")
        
        n = self.prediction.size
        gradient = 2.0 * (self.target - self.prediction) / n
        
        return gradient
