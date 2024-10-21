

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
        self.output_batch = self.neural_network.forward_pass(X)
        loss = self.cross_entropy_loss(Y, self.output_batch)
        print(f"Loss: {loss}")

