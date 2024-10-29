import numpy as np
from Layer import Layer



class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.input = None
        self.output = None

    def add_layer(self, layer):
        self.layers.append(layer) # Add a layer into the neural network
    
    def forward_pass(self, X):
        output = X # Base case 
        for layer in self.layers:
            output = layer.forward_propagation(output) # Carry out propogation through all layers
        self.output = output
        return output # Return the resultant column-vector of the neural network
    
    def make_prediction(self):
        output = self.input # Base case 
        for layer in self.layers:
            output = layer.forward_propagation(output) # Carry out propogation through all layers
        self.output = output
        return output # Return the resultant column-vector of the neural network
   