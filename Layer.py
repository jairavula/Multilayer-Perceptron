import numpy as np

class Layer:

    def __init__(self, n_input, n_output, activation = None):
        self.W = np.random.randn(n_output, n_input) * 0.01
        self.b = np.zeros((n_output, 1))
        self.activation = activation # Activation function

        self.Z = None # Z before applying activation
        self.A = None # A <- Z After applying activation

        self.training_Z = [] # Matrix to store Z vectors for batch training
        self.training_A = [] # Matrix to store A vectors for batch training
    
    # Rectified Linear Unit Function, Replace all values < 0 with 0
    def reLu(self, Z):
        return np.maximum(0,Z)

    # Z = activation_function(W * X + b)
    def forward_propagation(self, X):
        self.Z = np.dot(self.W, X) + self.b
        if self.activation is None:
            self.A = self.Z
        else:
            self.A = self.activation(self, self.Z)

        # Locally reshape Z and A for appending without modifying the original self.Z
        Z_to_append = np.atleast_2d(self.Z).reshape(-1, 1)
        A_to_append = np.atleast_2d(self.A).reshape(-1, 1)

        # Append the locally reshaped Z and A column vectors to the list
        self.training_Z.append(Z_to_append)
        self.training_A.append(A_to_append)

        return self.A

    # Activation function for output layer, converts raw vector values to probability distribution
    def softmax(self, Z):
        numerator = np.exp(Z)
        denominator = np.sum(np.exp(Z))
        return numerator / denominator

    def finalize_training_data(self):
        self.training_Z = np.hstack(self.training_Z)  # Convert list of arrays into a matrix
        self.training_A = np.hstack(self.training_A)  # Convert list of arrays into a matrix

        