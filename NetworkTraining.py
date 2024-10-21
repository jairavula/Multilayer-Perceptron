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

    

