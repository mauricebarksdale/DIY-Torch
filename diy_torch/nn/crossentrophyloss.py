import numpy as np
from diy_torch.nn.loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self):
        self.prediction = None
        self.target = None

    def forward(self, prediction, target):
        # For classification: prediction is (batch_size, num_classes), target is (batch_size,)
        if len(prediction.shape) != 2 or len(target.shape) != 1:
            raise ValueError("Prediction should be 2D (batch_size, num_classes), target should be 1D (batch_size,)")
        
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"Batch size mismatch: prediction {prediction.shape[0]} vs target {target.shape[0]}")

        self.prediction = prediction
        self.target = target

        # Convert logits to probabilites; -np.max prevents overflow
        exp_pred = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        softmax = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        batch_size = prediction.shape[0]
        correct_class_probability = softmax[np.arange(batch_size), self.target]

        # Add small epsilon to prevent log(0)
        loss = -np.mean(np.log(correct_class_probability + 1e-15))

        self.softmax = softmax
        return loss
    
    def backward(self):
        if not self.prediction or not self.target:
            raise RuntimeError("Must call forward() before backward()")
        
        batch_size = self.prediction.shape[0]

        grad = self.softmax.copy()

        grad[np.arange(batch_size), self.target] -= 1
        return grad / batch_size

