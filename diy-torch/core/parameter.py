import numpy as np

class Parameter:
    def __init__(self, data):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)