import numpy as np

class Layer:

    def __init__(self, n_input, n_output, activation = None):
        self.W = np.random.randn(n_output, n_input) * 0.01
        self.b = np.zeros((n_output, 1))
        self.activation = activation # Activation function

        self.Z = None # Z before applying activation
        self.A = None # A <- Z After applying activation

        self.training_Z = None # Matrix to store Z vectors for batch training
        self.training_A = None # Matrix to store A vectors for batch training

    
    # Rectified Linear Unit Function, Replace all values < 0 with 0
    def reLu(self, Z):
        return np.maximum(0,Z)

    # Z = activation_function(W * X + b)
    def forward_propagation(self, X):
        self.Z = np.dot(self.W, X) + self.b 

        if self.activation is None:
            self.A = self.Z # No activation function applied to Z
        else:
            self.A = self.activation(self, self.Z)

        # Reshape Z and A to be 2D column vectors and append to training data
        Z_to_append = np.atleast_2d(self.Z).reshape(-1, 1)  # Force 2D, column vector
        A_to_append = np.atleast_2d(self.A).reshape(-1, 1)  # Force 2D, column vector

        if self.training_Z is None:
            #  Initialize training matrices
            self.training_Z = Z_to_append 
            self.training_A = A_to_append
        else:
            # Append to training matrices
            self.training_Z = np.hstack((self.training_Z, Z_to_append))
            self.training_A = np.hstack((self.training_A, A_to_append))

        return self.A

    # Activation function for output layer, converts raw vector values to probability distribution
    def softmax(self, Z):
        numerator = np.exp(Z)
        denominator = np.sum(np.exp(Z))
        return numerator / denominator


        