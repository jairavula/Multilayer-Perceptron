# Neural Network from Scratch- WORK IN PROGRESS

## Overview

This project is an implementation of a basic **multi-layer perceptron (MLP)** neural network from scratch, without the use of high-level machine learning libraries like TensorFlow or PyTorch. It demonstrates a deep understanding of fundamental neural network concepts, including forward propagation, activation functions, softmax, and cross-entropy loss.

The goal of this project is to provide a hands-on, low-level implementation of a neural network, starting with manually coded matrix multiplication for weight updates, activation functions, and loss calculation.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Implemented Features](#implemented-features)
    - [Forward Propagation](#forward-propagation)
    - [Activation Functions](#activation-functions)
    - [Softmax and Cross-Entropy Loss](#softmax-and-cross-entropy-loss)
3. [Next Steps](#next-steps)
4. [How to Run](#how-to-run)

---

## Project Structure

```bash
.
├── NetworkTraining.py  # Training class to run forward pass and loss calculations
├── NeuralNetwork.py    # NeuralNetwork class with forward propagation
├── Layer.py            # Layer class with activation functions and weight matrices
├── Tests.py            # Unit tests for forward propagation and loss calculation
└── README.md           # Project README (this file)
```


## Implemented Features

### 1. Forward Propagation
The neural network processes the input data through multiple layers using matrix multiplication and adds a bias term at each layer. The forward propagation function is implemented for a network with the following structure:
- **Input Layer**: Takes the input features (e.g., 3 features).
- **Hidden Layer**: Applies ReLU activation with custom weight matrices and biases.
- **Output Layer**: Outputs predictions using softmax activation for multi-class classification.

Example flow:
1. Input data → Hidden layer (ReLU) → Output layer (Softmax).
2. Each layer is manually constructed with pre-defined weights and biases.

### 2. Activation Functions
The network uses different activation functions for different layers:
- **ReLU (Rectified Linear Unit)** for hidden layers:
    - \( \text{ReLU}(x) = \max(0, x) \)
- **Softmax** for the output layer, which converts raw scores to probabilities:
    - \( \text{Softmax}(z) = \frac{e^{z_i}}{\sum_j e^{z_j}} \)

### 3. Softmax and Cross-Entropy Loss
The **softmax function** is used in the output layer to convert the final layer's output into probabilities for multi-class classification. Afterward, the **cross-entropy loss** is calculated to determine the error between the predicted probabilities and the true one-hot encoded labels.

Example calculation:
- For an output \( Z \), the loss is computed as:
  \[
  L = -\frac{1}{N} \sum_{i=1}^{N} y_i \cdot \log(\hat{y_i})
  \]
  where \( \hat{y_i} \) is the predicted probability and \( y_i \) is the true label (one-hot encoded).


## Next Steps

### 1. Backpropagation
- Implement backpropagation to calculate the gradients of the loss with respect to the weights and biases for each layer.
- Use these gradients to update the weights during training using gradient descent.

### 2. Training Loop
- Build a training loop to perform multiple epochs of forward propagation, backpropagation, and weight updates.
- Add batch processing if needed and monitor training progress over time.

### 3. Validation and Testing
- Test the model on unseen data to check generalization performance.
- Track metrics such as accuracy or loss on validation/test datasets.






  




