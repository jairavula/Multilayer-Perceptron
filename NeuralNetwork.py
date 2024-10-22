import numpy as np
from Layer import Layer



class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer) # Add a layer into the neural network
    
    def forward_pass(self, X):
        output = X # Base case 
        for layer in self.layers:
            output = layer.forward_propagation(output) # Carry out propogation through all layers
        return output # Return the resultant column-vector of the neural network
   