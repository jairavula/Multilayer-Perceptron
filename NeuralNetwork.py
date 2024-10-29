import numpy as np
from Layer import Layer



class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.input = None
        self.output = None

    def add_layer(self, layer, index=None):
        if index is None:
            # Append if no index is specified
            self.layers.append(layer)
        else:
            # Insert at the specified index, shifting other layers down
            self.layers.insert(index, layer)

    def remove_layer(self, index):
        if 0 <= index < len(self.layers):
            # Remove the layer at the specified index
            removed_layer = self.layers.pop(index)
            print(f"Layer at index {index} removed: {removed_layer}")
        else:
            # Handle invalid index gracefully
            raise IndexError("Layer index out of range.")
    
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
   