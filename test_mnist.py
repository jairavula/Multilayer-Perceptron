import numpy as np
import pandas as pd
import kagglehub
import os
from Model import Model
from dotenv import load_dotenv
from Layer import Layer


def load_and_preprocess_data(data_path, filename):
    """Load and preprocess the MNIST CSV data."""
    # Load the CSV file
    data = pd.read_csv(os.path.join(data_path, filename))

    # Separate labels and features
    labels = data.iloc[:, 0].values  # First column is the label
    images = data.iloc[:, 1:].values  # Remaining columns are the pixel values

    # Normalize pixel values to the range [0, 1]
    images = images / 255.0

    # Reshape images to (784, num_images)
    images = images.T  # Transpose to get (784, num_images)

    # One-hot encode labels to create column vector format for each label
    num_classes = 10
    one_hot_labels = np.zeros((num_classes, labels.size))
    one_hot_labels[labels, np.arange(labels.size)] = 1  # Create one-hot encoding with shape (10, num_images)

    return images, one_hot_labels

def main():
    print("starting")
    load_dotenv()

    # Load dataset
    dataset_path = "mnist/"
    mnist_training_images, mnist_training_labels = load_and_preprocess_data(dataset_path, "mnist_train.csv")
    mnist_testing_images, mnist_testing_labels = load_and_preprocess_data(dataset_path, "mnist_test.csv")

   # Select a sample input from the test set
    sample_index = 0  # Change this index to view different examples
    sample_input = mnist_testing_images[:, sample_index].reshape(-1, 1)  # Select a single column as a column vector (784, 1)
    sample_label = mnist_testing_labels[:, sample_index].reshape(-1, 1)  # Corresponding one-hot encoded label as a column vector (10, 1)

    model = Model(784, 10, learning_rate= 0.01, batch_size= 64, epochs= 2)
    model.add_layer(784, 64, 0, activation= Layer.reLu)
    model.add_layer(64,10,1, activation= Layer.softmax)

    # Get model output for the untrained model
    untrained_output = model.process_raw_single_input(sample_input)

    print(f"Sample Label: {sample_label}")
    print(f"Untrained Model Output: {untrained_output}")

    # Train the model
    model.train_model(mnist_training_images, mnist_training_labels)

    # Get model output for the trained model
    trained_output = model.process_raw_single_input(sample_input)

    print(f"Sample Label: {sample_label} \n")
    print(f"Untrained Model Output: {untrained_output} \n")
    print(f"Trained Model Output: {trained_output} \n")

    model.test_model_performance(mnist_testing_images, mnist_testing_labels)



if __name__ == "__main__":
    main()