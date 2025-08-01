import numpy as np
from .typing_helpers import ArrayType

class Optimizer():
    def __init__(self, device_manager=None):
        self.device = device_manager
        self.xp = device_manager.xp if device_manager else np  # fallback to CPU by default

    def update_weights(self, layer):
        raise NotImplementedError("Weight updates must be implemented by subclasses")

class SGD(Optimizer):
    def __init__(self, eta, device_manager=None):
        self.eta = eta
        super().__init__(device_manager=device_manager)
        
    def update_weights(self, layer):
        if hasattr(layer, "weights") and hasattr(layer, "dW"):
            layer.weights -= self.eta * layer.dW
            layer.biases -= self.eta * layer.db
            
        elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
            layer._gamma -= self.eta*layer.dGamma
            layer._beta -= self.eta*layer.dBeta

        return
    
class Momentum(Optimizer):
    def __init__(self, eta, beta, device_manager=None):
        self.beta = beta
        self.eta = eta
        self.momentum_vector = {}
        super().__init__(device_manager=device_manager)

    def update_weights(self, layer):
        layer_id = id(layer)
        
        if layer_id not in self.momentum_vector:
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                self.momentum_vector[layer_id] = { 
                    "mW": self.xp.zeros_like(layer.weights),
                    "mb": self.xp.zeros_like(layer.biases)  
                }

            if hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
                self.momentum_vector[layer_id].update({
                    "mGamma": self.xp.zeros_like(layer._gamma),
                    "mBeta": self.xp.zeros_like(layer._beta)
                })

         
        
        if hasattr(layer, "weights") and hasattr(layer, "dW"):
            self.momentum_vector[layer_id]["mW"] = self.beta*self.momentum_vector[layer_id]["mW"] - self.eta*layer.dW
            self.momentum_vector[layer_id]["mb"] = self.beta*self.momentum_vector[layer_id]["mb"] - self.eta*layer.db
            layer.weights += self.momentum_vector[layer_id]["mW"]
            layer.biases += self.momentum_vector[layer_id]["mb"]
        
        elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
            self.momentum_vector[layer_id]["mGamma"] = self.beta*self.momentum_vector[layer_id]["mGamma"] - self.eta*layer.dGamma
            self.momentum_vector[layer_id]["mBeta"] = self.beta*self.momentum_vector[layer_id]["mBeta"] - self.eta*layer.dBeta
            layer._beta += self.momentum_vector[layer_id]["mBeta"]
            layer._gamma += self.momentum_vector[layer_id]["mGamma"]

        return

class Adam(Optimizer):
    def __init__(self, eta, beta1, beta2, device_manager=None):
        self.beta1 = beta1
        self.beta2 = beta2

        self.eta = eta
        self.first_moment = {}
        self.second_moment = {}
        self.timestep = {}

        super().__init__(device_manager=device_manager)

    def update_weights(self, layer):
        layer_id = id(layer)

        if layer_id not in self.timestep:
            self.timestep[layer_id] = 0
        
        self.timestep[layer_id] += 1
        t = self.timestep[layer_id]
        
        if layer_id not in self.first_moment:
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                self.first_moment[layer_id] = { 
                    "mW": self.xp.zeros_like(layer.weights),
                    "mb": self.xp.zeros_like(layer.biases)  
                }

                self.second_moment[layer_id] = {
                    "sW": self.xp.zeros_like(layer.weights),
                    "sb": self.xp.zeros_like(layer.biases)
                }

            if hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
                self.first_moment[layer_id].update({
                    "mGamma": self.xp.zeros_like(layer._gamma),
                    "mBeta": self.xp.zeros_like(layer._beta)
                })

                self.second_moment[layer_id] = {
                    "sGamma": self.xp.zeros_like(layer._gamma),
                    "sBeta": self.xp.zeros_like(layer._beta)
                }
        
        if hasattr(layer, "weights") and hasattr(layer, "dW"):
            self.first_moment[layer_id]["mW"] = self.beta1*self.first_moment[layer_id]["mW"] - (1 - self.beta1)*layer.dW
            self.first_moment[layer_id]["mb"] = self.beta1*self.first_moment[layer_id]["mb"] - (1 - self.beta1)*layer.db
            self.second_moment[layer_id]["sW"] = self.beta2*self.second_moment[layer_id]["sW"] + (1 - self.beta2)*(layer.dW)**2
            self.second_moment[layer_id]["sb"] = self.beta2*self.second_moment[layer_id]["sb"] + (1 - self.beta2)*(layer.db)**2

            m_hatw = self.first_moment[layer_id]["mW"]/(1 - self.beta1**t)
            m_hatb = self.first_moment[layer_id]["mb"]/(1 - self.beta1**t)
            s_hatw = self.second_moment[layer_id]["sW"]/(1 - self.beta2**t)
            s_hatb = self.second_moment[layer_id]["sb"]/(1 - self.beta2**t)

            layer.weights += self.eta*m_hatw/(self.xp.sqrt(s_hatw + 1e-6))
            layer.biases += self.eta*m_hatb/(self.xp.sqrt(s_hatb + 1e-6))
        
        elif hasattr(layer, "_gamma") and hasattr(layer, "_beta"):
            self.first_moment[layer_id]["mGamma"] = self.beta1*self.first_moment[layer_id]["mGamma"] - (1 - self.beta1)*layer.dGamma
            self.first_moment[layer_id]["mBeta"] = self.beta1*self.first_moment[layer_id]["mBeta"] - (1 - self.beta1)*layer.dBeta
            self.second_moment[layer_id]["sGamma"] = self.beta2*self.second_moment[layer_id]["sGamma"] + (1 - self.beta2)*(layer.dGamma)**2
            self.second_moment[layer_id]["sBeta"] = self.beta2*self.second_moment[layer_id]["sBeta"] + (1 - self.beta2)*(layer.dBeta)**2

            m_hatg = self.first_moment[layer_id]["mGamma"]/(1 - self.beta1**t)
            m_hat_beta = self.first_moment[layer_id]["mBeta"]/(1 - self.beta1**t)
            s_hatg = self.second_moment[layer_id]["sGamma"]/(1 - self.beta2**t)
            s_hat_beta = self.second_moment[layer_id]["sBeta"]/(1 - self.beta2**t)

            layer._gamma += self.eta*m_hatg/(self.xp.sqrt(s_hatg + 1e-6))
            layer._beta += self.eta*m_hat_beta/(self.xp.sqrt(s_hat_beta + 1e-6))

        return



        

        


