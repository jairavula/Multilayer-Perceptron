import numpy as np

class Training:

    def __init__(self, neural_network, learning_rate = 0.01):
        self.neural_network = neural_network
        self.output_batch = []
        self.learning_rate: float = learning_rate
        self.training_input_batch = None
        self.training_true_output = None

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon) # Replace 0's and 1's with close values

        loss = -np.sum(y_true * np.log(y_pred_clipped)) # Calculate Loss for all column-vectors in batch
        average_loss = loss / y_true.shape[1] # Find average loss based on batch size

        return average_loss
    
    def training_pass(self):
        self.output_batch = []

        # Loop through each example (each column in X)
        for i in range(self.training_input_batch.shape[1]):
            input_example = self.training_input_batch[:, i:i+1]  # Extract the i-th column as a 2D array (column vector)
            output = self.neural_network.forward_pass(input_example) 
            print(output)
            self.output_batch.append(output)  # Append to the output batch

        self.output_batch = np.hstack(self.output_batch)  # Stack column vectors into a matrix
        loss = self.cross_entropy_loss(self.training_true_output, self.output_batch) # Compute loss across entire batch

        return loss

    def output_layer_backpropagation(self):
        y_pred = self.output_batch
        error = y_pred - self.training_true_output  # Error matrix from processed batch

        # Get activations from the previous layer
        A_prev_layer = self.neural_network.layers[-2].training_A

        # Compute gradient of loss with respect to weights
        dW = np.dot(error, A_prev_layer.T) / self.training_input_batch.shape[1]

        # Compute gradient of loss with respect to biases
        dB = np.sum(error, axis=1, keepdims=True)

        # Save output layer weight and bias gradients
        self.neural_network.layers[-1].dW = dW 
        self.neural_network.layers[-1].dB = dB
        self.neural_network.layers[-1].error = error

        return dW, dB, error

    def hidden_layer_backpropagation(self, layer_idx):

        # Get current layer weights and training Z (pre-activations)
        W_layer = self.neural_network.layers[layer_idx + 1].W  # W from the next layer
        Z_current = np.array(self.neural_network.layers[layer_idx].training_Z)  # Convert to numpy arrray
    
        if layer_idx == 0:
            # If previous layer is the input, input matrix acts as activations
            A_previous = self.training_input_batch
        else:
            # Else, Get activations of the previous hidden layer
            A_previous = self.neural_network.layers[layer_idx - 1].training_A 

        A_previous_T = A_previous.T  # Transpose to match dimensions for matrix multiplication
        backpropogated_error = self.neural_network.layers[layer_idx + 1].error

        # Backpropagate the error from the next layer to the current layer
        error_current_raw = np.dot(W_layer.T, backpropogated_error)

        # Apply ReLU derivative to the error for the current layer
        relu_derivative = Z_current > 0  # Derivative of ReLU: 1 where Z > 0, 0 where Z <= 0
        error_current = error_current_raw * relu_derivative

        # Compute gradient of loss with respect to weights (dW)
        dW = np.dot(error_current, A_previous_T) 

        # Compute gradient of loss with respect to biases (dB)
        dB = np.sum(error_current, axis=1, keepdims=True)

        # Save hidden layer weight and bias gradients
        self.neural_network.layers[layer_idx].dW = dW
        self.neural_network.layers[layer_idx].dB = dB
        self.neural_network.layers[layer_idx].error = error_current

        # print(f"Z_current shape: {Z_current.shape}")
        # print(f"Z_current:\n{Z_current}\n")
        # print(f"Backpropogated error: \n {error}")
        # print(f"A_previous_T shape: {A_previous_T.shape}")
        # print(f"A_previous_T values:\n{A_previous_T}\n")
        # print(f"Current Error Raw shape: {error_current_raw.shape}")
        # print(f"Current Error Raw:\n{error_current_raw}\n")
        # print(f"Current Error shape: {error_current.shape}")
        # print(f"Current Error :\n{error_current}\n")

        return dW, dB, error_current

    # Network weight and bias updating via gradient descent
    def gradient_descent_update(self):

        for i, layer in enumerate(self.neural_network.layers):
            if layer.dW is not None and layer.dB is not None: 
                print(f"Previous Layer {i} weights: {layer.W}")
                print(f"Previous Layer {i} biases: {layer.b}")
                print(f"Stored Layer {i} dW: {layer.dW}")
                print(f"Stored Layer {i} dB: {layer.dB}")
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.dB

                print(f"Layer {i} weights updated: {layer.W}")
                print(f"Layer {i} biases updated: {layer.b}")
            else:
                print(f" Missing dW or dB at Layer {idx}")


        return

    # Network weight and bias updating via the Adam Optimizer
    def adam_optimizer_update(self):
        return








    

