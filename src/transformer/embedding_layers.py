from layers import Layer
from device_manager import DeviceManager
from typing_helpers import ArrayType

import numpy as np

class TokenEmbeddingLayer(Layer):
    def __init__(self, vocab_size:int, d_model:int, device_manager:DeviceManager=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device_manager = device_manager
        self.xp = device_manager.xp if device_manager else np
        self.weights = self.initialize_weights()
    

    def initialize_weights(self):
        return self.xp.random.randn(self.vocab_size, self.d_model) * (1 / self.xp.sqrt(self.d_model))
    
    def to(self, device:str):
        self.device_manager.set_device(device)
        self.xp = self.device_manager.xp

        if self.weights is not None:
            self.weights = self.xp.asarray(self.weights)

    def build(self):
        pass

    def activate(self):
        token_ids = self.prev_layer.activate()
        return self.weights[token_ids]*self.xp.sqrt(self.d_model)

    def backward(self, dA):

        pass

