import numpy as np
import math
from .typing_helpers import ArrayType

class LossFunction():
    def __init__(self, device_manager=None):
        self.device = device_manager
        self.xp = device_manager.xp if device_manager else np  # fallback to CPU by default

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError
    
    def derivative(self, y_pred, y_true):
        raise NotImplementedError

class MSELoss(LossFunction):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)
    
    def compute_loss(self, y_pred, y_true):
        return self.xp.mean((y_pred - y_true)**2)

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.shape[0]

class CrossEntropyLoss(LossFunction):
    def __init__(self, device_manager=None):
        super().__init__(device_manager=device_manager)

    def compute_loss(self, y_pred, y_true):
        eps = 1e-10
        correct_probs = y_pred[self.xp.arange(y_true.shape[0]), y_true]
        loss = -self.xp.mean(self.xp.log(correct_probs + eps))
        return loss
    
    def derivative(self, y_pred, y_true): # ASSUMES THAT SOFTMAX ACTIVATION IS BEING USED AND COMBINES DERIVATIVES OF ACTIVATION AND LOSS FUNCTION
        y_true_one_hot = self.xp.eye(y_pred.shape[1])[y_true]
        return (y_pred - y_true_one_hot) / y_true.shape[0]
    

    

