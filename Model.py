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

        for cycles in range(self.epochs + 1):

            self.training.clear_layer_activations()
            loss = self.training.training_pass()
            self.training.output_layer_backpropagation()

            for layer_index in range(len(self.training.neural_network.layers) - 2, -1, -1):
                self.training.hidden_layer_backpropagation(layer_index)

            self.training.gradient_descent_update()

            if cycles % 100 == 0:
                print(f"Epoch {cycles}/{self.epochs} - Loss: {loss}")
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"Runtime: {runtime:.6f} seconds")

    # Split data into batches
    def create_batches(x, y, batch_size):
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            yield x[start:end], y[start:end]

    def train_model(self, x_dataset, y_dataset):

        start_time = time.time()

        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}/{epochs}")

            # Shuffle dataset
            indices = np.arrange(len(x_dataset))
            np.random.shuffle(indices)
            x_subset = x_dataset[indices]
            y_subset = y_train[indices]

            for x_batch, y_batch in self.create_batches(x_dataset, y_dataset, self.batch_size):
                self.train_batch(x_subset, y_subset)

        print("Training complete.")
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime: {runtime:.6f} seconds")


            

