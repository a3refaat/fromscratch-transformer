import numpy as np

class Activation():

    def __call__(self, Z):
        raise NotImplementedError
    
    def derivative(self, Z):
        raise NotImplementedError
    

class ReLU(Activation):

    def __call__(self, Z):
        return np.maximum(0, Z)
    
    def derivative(self, Z):
        return (Z > 0)

class Sigmoid(Activation):

    def __call__(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def derivative(self, Z):
        sig = self(Z)
        return sig*(1 - sig)

class Softmax(Activation):

    def __call__(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True)) # e^z (normalized to prevent overflow)
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True) # softmax(Zi) = e^zi/sum(e^zj)
    
    def derivative(self, Z):
        raise NotImplementedError(
            "Softmax derivative is not implemented. Use softmax only with cross-entropy loss, "
            "which handles the gradient directly."
        )

class Tanh(Activation):

    def __call__(self, Z):
        return np.tanh(Z)
    
    def derivative(self, Z):
        t = np.tanh(Z)
        return 1 - t**2

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, Z):
        return np.where(Z > 0, Z, self.alpha * Z) # If Z > 0, return Z, else, return alpha*Z

    def derivative(self, Z):
        return np.where(Z > 0, 1, self.alpha) # Via power rule