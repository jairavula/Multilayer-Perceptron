import numpy as np

class Layer:
    def __init__(self, n_input, n_output, activation = None):

        self.W = np.random.randn(n_output, n_input) * 0.01
        self.b = np.zeros((n_output, 1))
        self.activation = activation
    
    # Rectified Linear Unit Function, Replace all values < 0 with 0
    def reLu(self, Z):
        return np.maximum(0,Z)

    # Z = activation_function(W * X + b)
    def forward_propagation(self, X):
        Z_vector = np.dot(self.W, X) + self.b
        if self.activation is None:
            return Z_vector
        else:
            return self.activation(self,Z_vector)
        