import numpy as np
import time
from NetworkTraining import Training
from NeuralNetwork import NeuralNetwork
from Layer import Layer

class Model:
    def __init__(self, input_size, output_size, learning_rate= 0.01, batch_size= None, training= None, epochs= 1000):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Build underlying neural network
        self.neural_network = NeuralNetwork()



        #TODO: Implement architecture as param to model class
        if training is None:
            self.training = Training(neural_network= self.neural_network, learning_rate= learning_rate)

        else:
            self.training = training

    # TODO: ADD RANDOMIZATION SUPPORT
    # Interface for adding layers to local neural network
    def add_layer(self, n_input, n_output, layer_index= None, activation = None):
        new_layer = Layer(n_input, n_output, activation= activation)
        self.neural_network.add_layer(new_layer, layer_index)

    # Interface for removing layer to local neural network
    def remove_layer(self, layer_index):
        self.neural_network.remove_layer(layer_index)


    def process_raw_single_input(self, raw_column_vector_input):
        self.neural_network.input = raw_column_vector_input
        raw_single_output = self.neural_network.make_prediction()
        return raw_single_output

    def train_batch(self, raw_column_vector_input_batch, raw_true_output_batch):

        start_time = time.time()

        self.training = Training(self.neural_network, self.learning_rate)

        self.training.training_input_batch = raw_column_vector_input_batch
        self.training.training_true_output = raw_true_output_batch


        self.training.clear_layer_activations()
        loss = self.training.training_pass()
        self.training.output_layer_backpropagation()

        for layer_index in range(len(self.training.neural_network.layers) - 2, -1, -1):
            self.training.hidden_layer_backpropagation(layer_index)

        self.training.gradient_descent_update()
        
        end_time = time.time()
        runtime = end_time - start_time
        return loss
        
        # print(f"Runtime: {runtime:.6f} seconds")

    def create_batches(self, x, y):
        """Yield successive mini-batches from the dataset."""
        for start in range(0, x.shape[1], self.batch_size):  # Iterate over columns
            end = start + self.batch_size
            
            # Ensure x_batch has shape (784, batch_size) for input
            x_batch = x[:, start:end].reshape(784, -1)  # Flatten and select batch columns
            
            # Ensure y_batch has shape (10, batch_size) for comparison
            y_batch = y[:, start:end]  # Select corresponding labels as columns
            
            yield x_batch, y_batch

    def train_model(self, x_dataset, y_dataset):
        start_time = time.time()

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}/{self.epochs}")

            # Iterate over mini-batches
            index = 0
            for x_batch, y_batch in self.create_batches(x_dataset, y_dataset):
                index += 1
                loss = self.train_batch(x_batch, y_batch) 
                if index % 100 == 0:
                    print(f"Batch {index} now processing... Loss: {loss}")

        print("Training complete.")
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime: {runtime:.6f} seconds")

    def test_model_performance(self, x_dataset, y_dataset):
        correct_predictions = 0
        total_predictions = x_dataset.shape[1]
        for i in range(total_predictions):
            output = self.neural_network.forward_pass(x_dataset[:, i].reshape(-1, 1))  # Ensure it's a column vector
            predicted_label = np.zeros_like(output)
            predicted_label[np.argmax(output)] = 1

            true_label= y_dataset[:, i].reshape(-1, 1)

            if np.array_equal(predicted_label, true_label):
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy of model's predictions: {accuracy:.4f}% ")


        # Convert the output to a 1 hot vector given the highest probability
        # Compare the two vectors, if equal, increment correct predicitions
        # Increment total predictions
        # Return correct over total

            

