import numpy as np
from .activations import ReLU, Sigmoid, Softmax, Tanh, LeakyReLU


ACTIVATION_MAP = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU
}

class Layer():
    def __init__(self, num_neurons=1, activation:bool=False, init_weights:bool=False, activation_function:str=None, init_method:str=None):
        self.valid_intializations = ["glorot", "he"]
        self.num_neurons = num_neurons
        self.prev_layer = None
        self.biases = np.zeros((self.num_neurons, 1))
        self.training = None
        self.built = False
    
        if activation:
            self.activation_function = activation_function
        
        if init_weights:
            self.init_method = init_method
        
    @property
    def activation_function(self):
        return self._activation_function
    
    @activation_function.setter
    def activation_function(self, activation_function:str) -> None:
        if activation_function is None:
            self._activation_function = ACTIVATION_MAP["relu"]()
        elif activation_function not in ACTIVATION_MAP.keys():
            raise ValueError(f"Invalid activation function: {activation_function}. Please select from {list(ACTIVATION_MAP.keys())}")
        else:
            self._activation_function = ACTIVATION_MAP[activation_function]()
        return
    
    def build(self):
        if self.built:
            return
        else:
            self.initialize_weights()
            self.built = True
        return
    
    def infer_fan_in(self):
        if isinstance(self.prev_layer, InputLayer):
            if self.prev_layer.flatten:
                if self.prev_layer.inputs is None:
                    raise ValueError("InputLayer must have inputs set before initializing weights.")
                fan_in = self.prev_layer.inputs.reshape(self.prev_layer.inputs.shape[0], -1).shape[1]
            else:
                fan_in = np.prod(self.prev_layer.input_shape)
        else:
            fan_in = self.prev_layer.num_neurons
        
        return fan_in

    def initialize_weights(self):
        fan_in = self.infer_fan_in()
        fan_out = self.num_neurons
        
        if self._init_method == "glorot":
            self.glorot_init(fan_in, fan_out)
        
        elif self._init_method == "he":
            self.he_init(fan_in, fan_out)

        return
    
    def glorot_init(self, fan_in, fan_out):
        self.weights = np.random.randn(fan_in, fan_out)*np.sqrt(2/fan_in + fan_out)
        return
    
    def he_init(self, fan_in, fan_out):
        self.weights = np.random.randn(fan_in, fan_out)*np.sqrt(2/fan_in)
        return
        
    @property
    def init_method(self):
        return self._init_method
    
    @init_method.setter
    def init_method(self, init_method:str):
        if init_method is None:
            self._init_method = "glorot"
        elif init_method not in self.valid_intializations:
            raise ValueError(f"Invalid weight initialization method: {init_method}. Please select from {self.valid_intializations}")
        else:
            self._init_method = init_method
        return
    
    def linear(self) -> np.ndarray:
        self.A_prev = self.prev_layer.activate() # Storing previous layer's activation ouput for backprop
        self.Z = np.dot(self.A_prev, self.weights) + self.biases.T # Also storing the current layer's linear output (inputs*weights)
        return self.Z
    
    def activate(self) -> np.ndarray:
        Z = self.linear()

        return self._activation_function(Z)
    
    def backward(self, dA:np.ndarray) -> np.ndarray:

        if isinstance(self._activation_function, Softmax):
            raise RuntimeError("Softmax derivative must be calculated by cross-entropy loss function.")

        dZ = dA*self._activation_function.derivative(self.Z)
        m = dZ.shape[0]

        self.dW = self.A_prev.T@dZ/m
        self.db = np.mean(dZ, axis=0).reshape(-1, 1)

        dA_prev = dZ@self.weights.T

        return self.prev_layer.backward(dA_prev)

class InputLayer:
    def __init__(self, input_shape:np.ndarray, flatten=False):
        self.inputs = None # Inputs are passed via neural network .fit()
        self.input_shape = input_shape
        self.flatten = flatten
        
    def activate(self) -> np.ndarray:
        if self.inputs is None:
            raise ValueError("No inputs found in input layer.")

        if self.flatten:
            return self.inputs.reshape(self.inputs.shape[0], -1)
        
        return self.inputs
    
    def backward(self, dA):
        return dA
    
class HiddenLayer(Layer):
    def __init__(self, num_neurons, activation_function:str=None, init_method:str=None):
        super().__init__(num_neurons=num_neurons, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        
class OutputLayer(Layer):
    def __init__(self, num_neurons, activation_function:str=None, init_method:str=None):
        super().__init__(num_neurons=num_neurons, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        self.next_layer = None
    
    def backward(self, dA):
        
        if isinstance(self._activation_function, Softmax):
            dZ = dA
        else:
            dZ = dA*self._activation_function.derivative(self.Z)
        
        m = dZ.shape[0]

        self.dW = self.A_prev.T@dZ/m
        self.db = np.mean(dZ, axis=0).reshape(-1, 1)

        dA_prev = dZ@self.weights.T

        return self.prev_layer.backward(dA_prev)

class BatchNormLayer(Layer):
    def __init__(self, num_neurons=1, activation = False, init_weights = False, activation_function = None, init_method = None, momentum = 0.9, eps=1e-5):
        super().__init__(num_neurons=num_neurons, activation=True, init_weights=True, activation_function=activation_function, init_method=init_method)
        self._gamma = np.ones(num_neurons,)
        self._beta = np.zeros(num_neurons,)
        self._running_mean = np.zeros(num_neurons,)
        self._running_var = np.ones(num_neurons,)
        self._momentum = momentum
        self.eps = eps
        
    
    def linear(self) -> np.ndarray:
        self.A_prev = self.prev_layer.activate() # Storing previous layer's activation ouput for backprop
        batch_size = self.A_prev.shape[0]
        self.Z = self.batch_norm(np.dot(self.A_prev, self.weights), batch_size)# Also storing the current layer's linear output (inputs*weights)

        return self.Z
    
    def batch_norm(self, Z, batch_size):
        eps = 1e-5

        if self.training:
            self.avg = np.mean(Z, axis=0)
            self.var = np.var(Z, axis=0)

            self._running_mean = self._momentum * self._running_mean + (1 - self._momentum) * self.avg # Running mean and variance updated for inference
            self._running_var = self._momentum * self._running_var + (1 - self._momentum) * self.var

            self.Z_norm = (Z - self.avg)/np.sqrt(self.var + eps)
        
        else:
            self.Z_norm = (Z - self._running_mean) / np.sqrt(self._running_var + self.eps) # Use running averages for inference

        return self._gamma*self.Z_norm + self._beta
    
    def backward(self, dA):
   
        dZ_out = dA
        m = dZ_out.shape[0]

        self.dGamma = np.sum(dZ_out *self.Z_norm, axis=0)
        self.dBeta = np.sum(dZ_out, axis=0)
        dZ_norm = dZ_out*self._gamma

        dVar = np.sum(dZ_norm * (self.Z - self.avg) * -0.5 / (self.var + self.eps)**1.5, axis=0)
        dMean = np.sum(dZ_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dVar * np.mean(-2 * (self.Z - self.avg), axis=0)

        dZ = dZ_norm / np.sqrt(self.var + self.eps) + dVar * 2 * (self.Z - self.avg) / m + dMean / m
        
        return self.prev_layer.backward(dZ)



