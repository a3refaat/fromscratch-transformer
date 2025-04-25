import numpy as np
from matplotlib import pyplot as plt
plt.ion()

from .losses import CrossEntropyLoss, MSELoss
from .layers import Layer, InputLayer, OutputLayer
from .optimizers import SGD, Momentum, Adam

LOSS_MAP = {
    'cross_entropy': CrossEntropyLoss,
    'mse': MSELoss,
}

OPTIMIZER_MAP = {
    'sgd': SGD,
    'momentum': Momentum,
    'adam' : Adam
}

class NeuralNetwork():
    def __init__(self, layers:list[Layer], epochs:int, eta:float, loss_func=None, optimizer=None):
        self.layers = layers
        self.epochs = epochs
        self.eta = eta
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.built = False
        self.__link_layers()

    def __link_layers(self):
        for i, layer in enumerate(self.layers):
            layer.prev_layer = self.layers[i - 1] if i > 0 else None
            layer.next_layer = self.layers[i + 1] if i < len(self.layers) - 1 else None
    
    def __build_layers(self):
        for layer in self.layers:
            if hasattr(layer, "build"):
                layer.build()
        self.built = True
        return

    @property
    def loss_func(self):
        return self._loss_func
    
    @loss_func.setter
    def loss_func(self, loss_func:str) -> None:
        if loss_func is None:
            self._loss_func = LOSS_MAP['mse']()
        elif loss_func not in LOSS_MAP.keys():
            raise ValueError(f"Invalid loss function: {loss_func}. Please select from {list(LOSS_MAP.keys())}")
        else:
            self._loss_func = LOSS_MAP[loss_func]()
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer) -> None:
        if optimizer is None:
            self._optimizer = SGD(eta=self.eta)
            ##FIX ME - ADD CHECK FOR INVALID OPTIMIZER
        else:
            self._optimizer = optimizer

    
    def __set_training(self, is_training:bool=True) -> None:
        for layer in self.layers:
            layer.training = is_training
        return
        
    def fit(self, X:np.ndarray, y:np.ndarray, batch_size:int, X_val=None, y_val=None, plot_curves=False): ## Optimizers to be added later
        self.train_losses = []
        self.validation_losses = []

        self.__set_training()

        for i in range(self.epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            batch_losses = []
            for j in range(0, len(X), batch_size):
                X_set = X[j:j + batch_size]
                y_set = y[j:j + batch_size]
                self.layers[0].inputs = X_set
                if not self.built:
                    self.__build_layers()

                y_pred = self.__forward_pass()
                loss = self.__compute_loss(y_pred, y_set)
                gradient = self.__backpropagate(y_pred, y_set)
                batch_losses.append(loss)
                self.__update_weights()

            epoch_loss = np.mean(batch_losses)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {i + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

            if X_val.any() and y_val.any():
                y_pred_val = self.predict(X_val)
                val_loss = self.__compute_loss(y_pred_val, y_val)
                self.validation_losses.append(val_loss)
            
            if plot_curves:
                self.__plot_learning_curves()
                
        plt.ioff()
        return
    
    def __plot_learning_curves(self):
        plt.clf()
        plt.plot(self.train_losses, label="Training Loss")

        if self.validation_losses:
            plt.plot(self.validation_losses, label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Live Learning Curves")
        plt.legend()
        plt.pause(0.001)

        return
        
    def __compute_loss(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        return self._loss_func.compute_loss(y_pred, y_true)

    def __forward_pass(self) -> np.ndarray:
        return self.layers[-1].activate()
    
    def __backpropagate(self, y_pred:np.ndarray, y_true:np.ndarray) -> np.ndarray:
        loss_derivative = self._loss_func.derivative(y_pred, y_true)
        return self.layers[-1].backward(loss_derivative)
    
    def __update_weights(self) -> None:
        for layer in self.layers:
            self._optimizer.update_weights(layer)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.layers[0].inputs = X
        self.__set_training(False)
        return self.__forward_pass()
        
