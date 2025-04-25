import numpy as np
import math

class LossFunction():

    def compute_loss(self, y_pred, y_true):
        raise NotImplementedError
    
    def derivative(self, y_pred, y_true):
        raise NotImplementedError

class MSELoss(LossFunction):

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class CrossEntropyLoss(LossFunction):

    def compute_loss(self, y_pred, y_true):
        eps = 1e-10
        correct_probs = y_pred[np.arange(y_true.shape[0]), y_true]
        loss = -np.mean(np.log(correct_probs + eps))
        return loss
    
    def derivative(self, y_pred, y_true): # ASSUMES THAT SOFTMAX ACTIVATION IS BEING USED AND COMBINES DERIVATIVES OF ACTIVATION AND LOSS FUNCTION
        y_true_one_hot = np.eye(y_pred.shape[1])[y_true]
        return (y_pred - y_true_one_hot) / y_true.shape[0]
    

    

