import numpy as np
from .typing_helpers import ArrayType

class Activation():
    def __init__(self, device_manager=None):
        self.device = device_manager
        self.xp = device_manager.xp if device_manager else np

    def __call__(self, Z):
        raise NotImplementedError
    
    def derivative(self, Z):
        raise NotImplementedError
    

class ReLU(Activation):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)

    def __call__(self, Z):
        return self.xp.maximum(0, Z)
    
    def derivative(self, Z):
        return (Z > 0)

class Sigmoid(Activation):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)

    def __call__(self, Z):
        return 1 / (1 + self.xp.exp(-Z))
    
    def derivative(self, Z):
        sig = self(Z)
        return sig*(1 - sig)

class Softmax(Activation):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)

    def __call__(self, Z):
        exp_Z = self.xp.exp(Z - self.xp.max(Z, axis=-1, keepdims=True)) # e^z (normalized to prevent overflow)
        return exp_Z / self.xp.sum(exp_Z, axis=-1, keepdims=True) # softmax(Zi) = e^zi/sum(e^zj)
    
    def derivative(self, Z):
        raise NotImplementedError(
            "Softmax derivative is not implemented. Use softmax only with cross-entropy loss, "
            "which handles the gradient directly."
        )

class Tanh(Activation):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)

    def __call__(self, Z):
        return self.xp.tanh(Z)
    
    def derivative(self, Z):
        t = self.xp.tanh(Z)
        return 1 - t**2

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01, device_manager=None):
        self.alpha = alpha
        super().__init__(device_manager=device_manager)

    def __call__(self, Z):
        return self.xp.where(Z > 0, Z, self.alpha * Z) # If Z > 0, return Z, else, return alpha*Z

    def derivative(self, Z):
        return self.xp.where(Z > 0, 1, self.alpha) # Via power rule