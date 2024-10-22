import unittest
import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from NetworkTraining import Training

def test_forward_propagation():
    
    input_layer = np.array([[1], [2], [3]]) # Column Vector input X (3 input neurons)
    second_layer = Layer(3,2, activation= Layer.reLu) # Hidden layer (3 inputs 2 output neurons)

    second_layer.W = np.array([[-0.2, 0.4, 0.6], [0.1, -0.5, 0.3]]) # Hidden Layer Weights
    second_layer.b = np.array([[0.5], [-0.2]]) # Hidden Layer Biases

    expected_output = np.array([[2.9], [0]]) # Z1, Z2 from Z = W * X + b

    test_output = second_layer.forward_propagation(input_layer)

    np.testing.assert_almost_equal(test_output, expected_output, decimal = 6)

def test_full_forward_propagation():
    # 3 layer network, 3-2-3, ReLu on hidden layer and softmax on output test

    input_layer = np.array([[1], [2], [3]]) # Column Vector input X (3 input neurons)

    hidden_layer = Layer(3,2, activation= Layer.reLu) # Hidden layer (3 inputs 2 output neurons)
    hidden_layer.W = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.6]]) # Hidden Layer Weights
    hidden_layer.b = np.array([[0.1], [-0.2]]) # Hidden Layer Biases

    output_layer = Layer(2,3, activation= Layer.softmax) # Output layer (2 inputs 3 output neurons)
    output_layer.W = np.array([[0.3, -0.2],[-0.5, 0.4],[0.2, 0.1]]) # Output Layer Weights
    output_layer.b = np.array([[0.05], [-0.05], [0.1]]) # Output Layer Biases

    expected_hidden_layer = np.array([[1.3],[2.3]])
    expected_output_layer = np.array([[0.243], [0.309], [0.448]])

    test_hidden_layer = hidden_layer.forward_propagation(input_layer) # Input Layer -> Hidden Layer
    test_output_layer = output_layer.forward_propagation(test_hidden_layer) # Hidden Layer -> Output Layer

    np.testing.assert_almost_equal(test_hidden_layer, expected_hidden_layer, decimal = 3)
    np.testing.assert_almost_equal(test_output_layer, expected_output_layer, decimal = 3)

def test_forward_training_pass():

    network = NeuralNetwork() # Create a network

    hidden_layer = Layer(3, 2, activation = Layer.reLu) # Initialize hidden layer structure
    hidden_layer.W = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.6]]) # Hidden Layer Weights
    hidden_layer.b = np.array([[0.1], [-0.2]]) # Hidden Layer Biases


    output_layer = Layer(2, 3, activation = Layer.softmax) # Initialize output layer structure
    output_layer.W = np.array([[0.3, -0.2],[-0.5, 0.4],[0.2, 0.1]]) # Output Layer Weights
    output_layer.b = np.array([[0.05], [-0.05], [0.1]]) # Output Layer Biases

    network.add_layer(hidden_layer)
    network.add_layer(output_layer)

    # Network is now constructed, create a training instance
    training = Training(network)

    input_data = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]]) # Ex. 1 [0.1, 0.2, 0.3] Ex. 2 [0.5, 0.6, 0.7]
    true_output = np.array([[1, 0], [0, 1], [0, 0]]) # Ex. 1 [1, 0, 0] Ex. 2 [0, 1, 0]

    loss = training.training_pass(input_data, true_output)

    expected_loss = 1.1645417

    np.testing.assert_almost_equal(loss, expected_loss, decimal = 6)

def test_output_layer_backpropagation():

    network = NeuralNetwork() # Create a network

    hidden_layer = Layer(3, 2, activation = Layer.reLu) # Initialize hidden layer structure
    hidden_layer.W = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.6]]) # Hidden Layer Weights
    hidden_layer.b = np.array([[0.1], [-0.2]]) # Hidden Layer Biases


    output_layer = Layer(2, 3, activation = Layer.softmax) # Initialize output layer structure
    output_layer.W = np.array([[0.3, -0.2],[-0.5, 0.4],[0.2, 0.1]]) # Output Layer Weights
    output_layer.b = np.array([[0.05], [-0.05], [0.1]]) # Output Layer Biases

    network.add_layer(hidden_layer)
    network.add_layer(output_layer)

    # Network is now constructed, create a training instance
    training = Training(network)

    input_data = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]]) # Ex. 1 [0.1, 0.2, 0.3] Ex. 2 [0.5, 0.6, 0.7]
    true_output = np.array([[1, 0], [0, 1], [0, 0]]) # Ex. 1 [1, 0, 0] Ex. 2 [0, 1, 0]

    loss = training.training_pass(input_data, true_output)

    dW_matrix, dB_matrix, error = training.output_layer_backpropagation(true_output)
    print("dW (Output to Hidden):", dW_matrix)
    print("Error propagated to hidden layer:", error)

    dw2, db2, current_error = training.hidden_layer_backpropagation(1, error)
    print("dW (Input to Hidden):", dw2)  # Should print the weight gradients for the hidden layer
    print("dB (Hidden Layer Bias Gradients):", db2)



if __name__ == '__main__':
    test_forward_propagation()
    test_full_forward_propagation()
    test_forward_training_pass()
    test_output_layer_backpropagation()