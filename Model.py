import numpy as np
from NetworkTraining import Training
from NeuralNetwork import NeuralNetwork
from Layer import Layer

class Model:
    def __init__(self, input_size, output_size, learning_rate= 0.01, batch_size= None, training_architecture= None, epochs= 100):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Build underlying neural network
        self.neural_network = NeuralNetwork()



        #TODO: Implement architecture as param to model class
        if training_architecture is None:
            self.training_architecture = Training(neural_network= self.neural_network, learning_rate= learning_rate)

        else:
            self.training_architecture = training_architecture

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

    def train_model(self, raw_column_vector_input_batch, raw_true_output_batch):

        self.training_architecture = Training(self.neural_network, self.learning_rate)

        self.training_architecture.training_input_batch = raw_column_vector_input_batch
        self.training_architecture.training_true_output = raw_true_output_batch

        for cycles in range(self.epochs + 1):

            self.training_architecture.clear_layer_activations()
            loss = self.training_architecture.training_pass()
            self.training_architecture.output_layer_backpropagation()

            for layer_index in range(len(self.training_architecture.neural_network.layers) - 2, -1, -1):
                self.training_architecture.hidden_layer_backpropagation(layer_index)

            self.training_architecture.gradient_descent_update()

            if cycles % 10 == 0:
                print(f"Epoch {cycles}/{self.epochs} - Loss: {loss}")

