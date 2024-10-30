# Neural Network Library from Scratch- WORK IN PROGRESS



## Overview

This project is an implementation of a neural network/ basic ML model library. it implements a scalabe and configureable **multi-layer perceptron (MLP)** neural network architecture from scratch, without the use of high-level machine learning libraries like TensorFlow or PyTorch. The goal of this project is to provide a hands-on, low-level implementation of a neural network library that provides the essential components needed to build, train, and evaluate basic machine learning models. As development continues, more features and configureability will be added to the library, aiming to support more and more network architectures and training configurations.

![Alt Text](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20230602113310/Neural-Networks-Architecture.png)


---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Simple Library Example Usage](#example-usage)
3. [Implemented Features](#implemented-features)
    - [Forward Propagation](#forward-propagation)
    - [Activation Functions](#activation-functions)
    - [Softmax and Cross-Entropy Loss](#softmax-and-cross-entropy-loss)
    - [Backpropagation and Training Loop](#backpropagation-and-training-loop)
4. [Next Steps](#next-steps)
5. [How to Run](#how-to-run)

---

## Project Structure

```bash
.
├── Model.py  # Neural Network Model interface
├── NetworkTraining.py  # Training class to tune model
├── NeuralNetwork.py    # NeuralNetwork class holding raw model
├── Layer.py            # Layer class holding raw neuron layers 
├── Tests.py            # Unit tests for library implementation
└── README.md           # Project README (this file)
```

## Simple Library Example Usage

## Example Usage

Below is an example demonstrating how to create a model, add layers, and train it on an identity mapping task. We are training the neural network to output the input it receives:

```python

# Initialize Model with 4 input and 4 output nodes
model = Model(4, 4)

# Add the first hidden layer with ReLU activation
model.add_layer(4, 4, 0, activation=Layer.reLu)

# Add the output layer with Softmax activation
model.add_layer(4, 4, 1, activation=Layer.softmax)

# Define the input batch - an identity matrix for testing
input_batch = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])

# Define the expected output batch - also an identity matrix
true_output_batch = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0], 
                              [0, 0, 1, 0], 
                              [0, 0, 0, 1]])

# Check the model's output before training
untrained_output = model.process_raw_single_input(np.array([[1], [0], [0], [0]]))
print(f"Untrained Model Input: \n {model.neural_network.input}")
print(f"Untrained Model Output: \n {untrained_output}")

# Train the model on the input and output batches
model.train_model(input_batch, true_output_batch)

# Check the model's output after training
trained_output = model.process_raw_single_input(np.array([[1], [0], [0], [0]]))
print(f"Trained Model Input: \n {model.neural_network.input}")
print(f"Trained Model Output: \n {model.training_architecture.neural_network.output}")
```


## Table of Contents
1. [Project Structure](#project-structure)
2. [Simple Library Example Usage](#example-usage)
3. [Implemented Features](#implemented-features)
    - [Forward Propagation](#forward-propagation)
    - [Activation Functions](#activation-functions)
    - [Softmax and Cross-Entropy Loss](#softmax-and-cross-entropy-loss)
    - [Backpropagation and Training Loop](#backpropagation-and-training-loop)
4. [Next Steps](#next-steps)
5. [How to Run](#how-to-run)

---

## Implemented Features

### Forward Propagation
The neural network processes the input data through multiple layers using matrix multiplication and adds a bias term at each layer. The forward propagation function is implemented for a network with the following structure:
- **Input Layer**: Takes the input features (e.g., 3 features).
- **Hidden Layer**: Applies ReLU activation with custom weight matrices and biases.
- **Output Layer**: Outputs predictions using softmax activation for multi-class classification.

Example flow:
1. Input data → Hidden layer (ReLU) → Output layer (Softmax).
2. Each layer is manually constructed with pre-defined weights and biases.

### Activation Functions
The network uses different activation functions for different layers:
- **ReLU (Rectified Linear Unit)** for hidden layers:
    - `ReLU(x) = max(0, x)`
- **Softmax** for the output layer, which converts raw scores to probabilities:
    - `Softmax(z) = exp(z_i) / sum(exp(z_j))`

### Softmax and Cross-Entropy Loss
The **softmax function** is used in the output layer to convert the final layer's output into probabilities for multi-class classification. Afterward, the **cross-entropy loss** is calculated to determine the error between the predicted probabilities and the true one-hot encoded labels.

Example Loss Calculation:
For an output `Z`, the cross-entropy loss is computed as:
  - `L = -(1/N) * sum(y_i * log(y_hat_i))`
where `y_hat_i` is the predicted probability and `y_i` is the true label (one-hot encoded).

### Backpropagation and Training Loop
The **backpropagation algorithm** is a method for training neural networks by minimizing the loss function with respect to each parameter. By calculating gradients for each weight and bias in the network, backpropagation adjusts these values to reduce the output prediction error, gradually "learning" from the data.

In backpropagation, we compute:

1. **Error at the Output Layer**:
   - The first step in backpropagation is to calculate the error at the output layer. This error measures the difference between the network’s predictions (`y_hat`) and true values (`y`). For multi-class classification, the error (δ, or "delta") at the output layer is computed as:
     
     ```
     δ = y_hat - y
     ```
   - After obtaining `y_hat` and `y`, this difference serves as the starting point to propagate the error backward through the network.

2. **Gradients of Weights and Biases**:
   - Next, we compute gradients for each parameter (weights and biases) in each layer to determine each parameter's influence on the loss. Gradients represent partial derivatives of the loss function with respect to each parameter and are used to adjust the parameters to reduce the loss.

    For a given layer, the following formulas are used to calculate the gradients:
   - **Weight Gradient**:
     ```
     dW = (1 / N) * δ * A_prev^T
     ```
     Here, `A_prev` is the activation from the previous layer, and `N` is the number of samples. `A_prev^T` denotes the transpose of `A_prev` for correct matrix multiplication.
     
    - **Bias Gradient**:
     ```
     db = (1 / N) * sum(δ)
     ```
    Summing over δ aggregates the impact of each example in the batch on the bias gradient.
    
   - **Propagating Error to Previous Layer**:
      ```
      δ_prev = (W^T * δ) ∘ g'(Z)
      ```
      For each hidden layer, the new δ is calculated by applying the chain rule.

3. **Gradient Descent**:
   - After calculating all the gradients, we update the weights and biases for each layer using gradient descent. With each parameter update, the model "learns" by making small adjustments to minimize the loss:
     ```
     W := W - alpha * dW
     b := b - alpha * db
     ```
     Here, `alpha` (learning rate) controls the magnitude of each update.

#### Training Loop
The training loop performs:
1. **Forward Propagation** to generate predictions.
2. **Backpropagation** to calculate gradients.
3. **Parameter Update** using gradient descent.

This loop, repeated over multiple epochs, gradually reduces the loss, improving the model’s accuracy.

## Next Steps

To further enhance the functionality and flexibility of this neural network library, here are the next steps planned for development:

1. **Support for Additional Optimization Strategies**:
   - Implement optimizers such as **Adam** and **RMSprop** for more robust training.
   - These optimizers will adapt the learning rate for each parameter, improving training stability and convergence speed.

2. **Convolutional Neural Network (CNN) Support**:
   - Expand the library to include **convolutional layers** and **pooling layers**, enabling the creation of CNN architectures for tasks involving image data.
   - This will involve creating specialized `ConvLayer` and `PoolingLayer` classes with functionality for forward and backward propagation.

3. **Hyperparameter Flexibility**:
   - Enable users to customize more hyperparameters, such as **batch size**, **learning rate schedule**, **weight initialization**, and **activation functions**.
   - Allow users to configure layers with different types of initialization (e.g., Xavier, He initialization).

4. **Regularization Techniques**:
   - Add support for **dropout** and **L2 regularization** to improve generalization and prevent overfitting, especially in deeper networks.
   - Dropout will allow users to specify dropout rates per layer, randomly setting some activations to zero during training.

5. **Expanded Activation Functions**:
   - Include additional activations like **Leaky ReLU**, **Tanh**, and **Sigmoid** to provide more flexibility in configuring neural network architectures.

---

## How to Run

This project is structured as a library for building, training, and evaluating custom neural networks. To use this library in your own projects, follow these instructions:

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/neural-network-library.git
cd neural-network-library
```
```python

# Install required dependencies (numpy only for now)
pip install -r requirements.txt

# Import necessary modules to define, configure, and train your neural network
from neural_network_library import Model, Layer, Training

# Ready to use! Detailed interface documentation coming soon!
```










