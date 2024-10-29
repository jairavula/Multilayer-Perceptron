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
    print("dW (Output to Hidden): \n", dW_matrix)
    print("dB (Output to Hidden): \n", dB_matrix)
    print("Error propagated to hidden layer:", error)

    dw2, db2, current_error = training.hidden_layer_backpropagation(0, error, input_data)
    print("dW (Input to Hidden): \n", dw2)  # Should print the weight gradients for the hidden layer
    print("dB (Hidden Layer Bias Gradients):\n", db2)

def test_first_hidden_layer_activations():
    # Initialize the network with one hidden layer
    network = NeuralNetwork()  # Create a network

    input_matrix = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])  # 3 input neurons

    hidden_layer = Layer(3, 2, activation=Layer.reLu)  # 3 inputs, 2 outputs (hidden layer)
    hidden_layer.W = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.6]])  # Example weights
    hidden_layer.b = np.array([[0.1], [-0.2]])  # Example biases
    network.add_layer(hidden_layer)

    output_layer = Layer(2, 3, activation=Layer.softmax)  # 2 inputs, 3 outputs (output layer)
    output_layer.W = np.array([[0.3, -0.2], [-0.5, 0.4], [0.2, 0.1]])  # Example weights
    output_layer.b = np.array([[0.05], [-0.05], [0.1]])  # Example biases
    network.add_layer(output_layer)
    true_output = np.array([[1, 0], [0, 1], [0, 0]]) # Ex. 1 [1, 0, 0] Ex. 2 [0, 1, 0]
    training = Training(network)
    _ = training.training_pass(input_matrix, true_output )  # Forward pass without Y

    # Get activations of the first hidden layer (training_A)
    A_previous = network.layers[0].training_A  # This should now be the activations

    print("Shape of input_matrix:", input_matrix.shape)
    print("Shape of training_A (first hidden layer):", A_previous.shape)
    print("Values of input_matrix:\n", input_matrix)
    print("Values of training_A (first hidden layer):\n", A_previous)

    # Check if the activations shape matches the expected shape (2, 2) for 2 hidden neurons and 2 examples
    expected_shape = (2, input_matrix.shape[1])  # (2 hidden neurons, 2 examples)
    assert A_previous.shape == expected_shape, f"Expected shape {expected_shape}, but got {A_previous.shape}"

def test_multi_layer_network():

    network = NeuralNetwork()

    hidden_layer_1 = Layer(3,2, activation= Layer.reLu)
    hidden_layer_1.W = np.array([[0.1, -0.2, -0.3], [0.4, 0.5, 0.6]])
    hidden_layer_1.b = np.array([[0.5],[0.25]])

    hidden_layer_2 = Layer(2,3, activation= Layer.reLu)
    hidden_layer_2.W = np.array([[1.2, -1.4], [0.6, 0.9], [-2.3,4.1]])
    hidden_layer_2.b = np.array([[-1],[0.4],[-2.2]])

    output_layer = Layer(3,2, activation= Layer.softmax)
    output_layer.W = np.array([[0.7, -0.3, 1.1], [0.2,-1,-0.3]])
    output_layer.b = np.array([[0.72],[0.78]])

    network.add_layer(hidden_layer_1)
    network.add_layer(hidden_layer_2)
    network.add_layer(output_layer)


    training = Training(network)

    training.training_input_batch = np.array([[1, -0.5],[-2, 1],[3, 1.5]])
    training.training_true_output = np.array([[1,1],[0,0]])
    loss = training.training_pass()
    print(loss)

    dW_matrix, dB_matrix, error = training.output_layer_backpropagation()
    dw2, db2, current_error = training.hidden_layer_backpropagation(1)
    dw3, db3, final_error = training.hidden_layer_backpropagation(0)

    training.gradient_descent_update()
    


def test_identity_mapping_network():
    network = NeuralNetwork()

    hidden_layer = Layer(4,4, activation= Layer.reLu)
    hidden_layer.W = np.array([[0.1, -0.2, -0.3, -0.1], 
                                [0.4, 0.5, 0.6, -0.2], 
                                [0.1, -0.5, 0.2, -0.8], 
                                [0.1, 2.4, 1.3, -0.2]])
    hidden_layer.b = np.array([[0.5],[0.25], [3.1], [0.11]])

    output_layer = Layer(4,4, activation= Layer.softmax)
    output_layer.W = np.array([[1.1, -0.44, -0.3, -0.22], 
                                [0.76, -0.5, 0.32, -0.8], 
                                [2.4, -1.2, -0.27, -0.1], 
                                [0.1, 1.4, 1.3, -0.77]])
    output_layer.b = np.array([[1.2],[0.5], [3.1], [1.33]])

    network.add_layer(hidden_layer)
    network.add_layer(output_layer)

    training = Training(network)
    training.learning_rate = 0.1

    training.training_input_batch = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]])

    training.training_true_output = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]])

     # Run multiple epochs to observe learning progress
    for epoch in range(100):  # Adjust number as needed
        training.clear_layer_activations() # Clear previous activations
        loss = training.training_pass()  # Forward pass and calculate loss
        training.output_layer_backpropagation()  # Backprop for output layer
        dw2, db2, current_error = training.hidden_layer_backpropagation(0)  # Backprop for hidden layer
        training.gradient_descent_update()  # Update weights and biases

        # Print progress every few epochs
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch} - Loss: {loss}")

    







if __name__ == '__main__':
    # test_forward_propagation()
    # test_full_forward_propagation()
    # test_forward_training_pass()
    # test_output_layer_backpropagation()
    # test_first_hidden_layer_activations()
    # test_multi_layer_network()
    test_identity_mapping_network()