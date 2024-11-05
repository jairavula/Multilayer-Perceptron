import numpy as np

class Layer:

    def __init__(self, n_input, n_output, activation = None, initialization="he"):
        self.W = np.random.randn(n_output, n_input) * 0.01
        self.b = np.zeros((n_output, 1))
        self.activation = activation # Activation function

        self.Z = None # Z before applying activation
        self.A = None # A <- Z After applying activationsourcve

        self.training_Z = None # Matrix to store Z vectors for batch training
        self.training_A = None # Matrix to store A vectors for batch training

        self.dW = None # Stores gradient matrix for layer weights in backpropogation
        self.dB = None # Stores gradient matrix for layer biases in backpropogation
        self.error = None # Stores Error at layer to backpropogate

         # Random weight initialization
        if initialization == "xavier":
            limit = np.sqrt(6 / (n_input + n_output))
            self.W = np.random.uniform(-limit, limit, (n_output, n_input))
        elif initialization == "he":
            limit = np.sqrt(2 / n_input)
            self.W = np.random.normal(0, limit, (n_output, n_input))
        else:
            # No technique specified
            self.W = np.random.uniform(-0.01, 0.01, (n_output, n_input))

        # Bias initialization
        self.b = np.zeros((n_output, 1))

    

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

    # Rectified Linear Unit Function, Replace all values < 0 with 0
    def reLu(self, Z):
        return np.maximum(0,Z)

    # Activation function- Softmax, converts output to probability distribution
    def softmax(self, Z):
        numerator = np.exp(Z)
        denominator = np.sum(np.exp(Z))
        return numerator / denominator
    
    # Sigmoid function, use for binary classification
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))


        