import numpy as np

class Training:

    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.output_batch = []

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon) # Replace 0's and 1's with close values

        loss = -np.sum(y_true * np.log(y_pred_clipped)) # Calculate Loss for all column-vectors in batch
        average_loss = loss / y_true.shape[1] # Find average loss based on batch size

        return average_loss
    
    def training_pass(self, X, Y):
        self.output_batch = []

        # Loop through each example (each column in X)
        for i in range(X.shape[1]):
            input_example = X[:, i:i+1]  # Extract the i-th column as a 2D array (column vector)
            output = self.neural_network.forward_pass(input_example) 
            self.output_batch.append(output)  # Append to the output batch

        self.output_batch = np.hstack(self.output_batch)  # Stack column vectors into a matrix
        loss = self.cross_entropy_loss(Y, self.output_batch) # Compute loss across entire batch

        return loss

    def output_layer_backpropagation(self, Y):
        y_pred = self.output_batch
        error = y_pred - Y # Resultant error matrix from processed batch

        self.neural_network.layers[-2].finalize_training_data()

        A_prev_layer = self.neural_network.layers[-2].training_A.T # Previous layer neuron activations

        # # Computes gradient of loss with respect to weights for each neuron in layer
        dW = np.dot(error, A_prev_layer)

        # Computes gradient of loss with respect to biases for each neuron in layer
        dB = np.sum(error, axis = 1, keepdims= True)


        return  dW, dB, error

    def hidden_layer_backpropagation(self, layer_idx, error):

        # Current layer weights
        W_layer = self.neural_network.layers[layer_idx].W  
        Z_current = np.array(self.neural_network.layers[layer_idx].training_Z)

        # Get the activations and pre-activation values (Z) of the previous layer
        A_previous = np.array(self.neural_network.layers[layer_idx - 1].training_A) 

        # Backpropagate the error from the next layer to the current layer
        error_current_raw = np.dot(W_layer.T, error)

        # Apply the ReLU derivative to the error for the current layer
        relu_derivative = Z_current > 0  # Derivative of ReLU: 1 where Z > 0, 0 where Z <= 0
        print(relu_derivative)

        
        error_current = error_current_raw * relu_derivative 

         # Computes gradient of loss with respect to weights for each neuron in layer
        dW_layer = np.dot(error_current, A_previous.T)

        # Computes gradient of loss with respect to biases for each neuron in layer
        dB_layer = np.sum(error_current, axis=1, keepdims=True)

        return dW_layer, dB_layer, error_current  





    

