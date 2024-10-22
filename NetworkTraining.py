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
            print(output)
            self.output_batch.append(output)  # Append to the output batch

        self.output_batch = np.hstack(self.output_batch)  # Stack column vectors into a matrix
        loss = self.cross_entropy_loss(Y, self.output_batch) # Compute loss across entire batch

        return loss

    def output_layer_backpropagation(self, Y):
        y_pred = self.output_batch
        error = y_pred - Y  # Error matrix from processed batch

        # Get activations from the previous layer
        A_prev_layer = self.neural_network.layers[-2].training_A

        # Compute gradient of loss with respect to weights
        dW = np.dot(error, A_prev_layer.T)

        # Compute gradient of loss with respect to biases
        dB = np.sum(error, axis=1, keepdims=True)

        return dW, dB, error

    def hidden_layer_backpropagation(self, layer_idx, error, input_matrix = None):

        # Get current layer weights and training Z (pre-activations)
        W_layer = self.neural_network.layers[layer_idx + 1].W  # W from the next layer
        Z_current = np.array(self.neural_network.layers[layer_idx].training_Z)  # Ensure it's a NumPy array
        
    
        if layer_idx == 0:
            # If previous layer is the input, input matrix acts as activations
            A_previous = input_matrix  
        else:
            # Get activations of the previous hidden layer
            A_previous = self.neural_network.layers[layer_idx - 1].training_A 


        A_previous_T = A_previous.T  # Transpose to match dimensions for matrix multiplication
        print(f"A_previous_T shape: {A_previous_T.shape}")
        print(f"A_previous_T values:\n{A_previous_T}\n")

        # Backpropagate the error from the next layer to the current layer
        error_current_raw = np.dot(W_layer.T, error)

        # Apply ReLU derivative to the error for the current layer
        relu_derivative = Z_current > 0  # Derivative of ReLU: 1 where Z > 0, 0 where Z <= 0
        error_current = error_current_raw * relu_derivative

        # Compute gradient of loss with respect to weights (dW)
        dW_layer = np.dot(error_current, A_previous_T) 

        # Compute gradient of loss with respect to biases (dB)
        dB_layer = np.sum(error_current, axis=1, keepdims=True)

        return dW_layer, dB_layer, error_current










    

