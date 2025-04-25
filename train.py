import numpy as np

from sklearn.datasets import fetch_openml
from src import NeuralNetwork, InputLayer, HiddenLayer, OutputLayer, BatchNormLayer, SGD, Momentum, Adam
from visualize import NeuralNetworkVisualizer


if __name__ == "__main__":

    layers = [InputLayer(input_shape=[28, 28], flatten=True),
                BatchNormLayer(50, activation_function="leaky_relu", init_method="he"),
                BatchNormLayer(50, activation_function="leaky_relu", init_method="he"),
                BatchNormLayer(50, activation_function="leaky_relu", init_method="he"),
                HiddenLayer(50, activation_function="leaky_relu", init_method="he"),
                HiddenLayer(50, activation_function="leaky_relu", init_method="he"),
                HiddenLayer(50, activation_function="leaky_relu", init_method="he"),
                OutputLayer(10, activation_function="softmax", init_method="he")]
                             
    mnist = fetch_openml("mnist_784", version=1, parser='pandas')

    X = mnist.data.to_numpy()  # Convert to NumPy array
    X = X / 255.0 # Scale pixel values
    y = mnist.target.to_numpy().astype(int)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    

    momentum_optimizer = Momentum(eta=5e-3, beta=0.8)
    adam = Adam(eta=1e-3, beta1=0.9, beta2=0.999)

    neural_net = NeuralNetwork(layers, epochs=100, eta=1e-5, loss_func="cross_entropy", optimizer=adam)
    neural_net.fit(X_train, y_train, batch_size=32, X_val=X_test, y_val=y_test, plot_curves=True)

    
