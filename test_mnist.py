import numpy as np
import pandas as pd
import kagglehub
import os
from Model import Model
from dotenv import load_dotenv

def load_and_preprocess_data(data_path, filename):
    """Load and preprocess the MNIST CSV data."""
    # Load the CSV file
    data = pd.read_csv(os.path.join(data_path, filename))

    # Separate labels and features
    labels = data.iloc[:, 0].values  # First column is the label
    images = data.iloc[:, 1:].values  # Remaining columns are the pixel values

    # Normalize pixel values to the range [0, 1]
    images = images / 255.0

    # Reshape images to (num_samples, 28, 28) if the model expects 2D images
    images = images.reshape(-1, 28, 28)

     # One-hot encode labels to create column vector format for each label
    num_classes = 10
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    one_hot_labels = one_hot_labels.T  # Transpose to match (num_classes, num_samples)

    return images, one_hot_labels

def main():
    print("starting")
    load_dotenv()

    dataset_path = "mnist/"
    mnist_training_images, mnist_training_labels = load_and_preprocess_data(dataset_path, "mnist_train.csv")
    mnist_testing_images, mnist_testing_labels = load_and_preprocess_data(dataset_path, "mnist_train.csv")

    model = Model(784, 10, learning_rate= 0.01, batch_size= 64, epochs= 1000)

    # Select a sample input from the test set
    sample_index = 0  # You can change this index to view different examples
    sample_input = mnist_testing_images[sample_index].reshape(-1, 1)  # Reshape for column vector
    sample_label = mnist_testing_labels[:, sample_index]  # Corresponding one-hot encoded label
    
    # Get model output for the untrained model
    untrained_output = model.process_raw_single_input(sample_input)
    print("Before Training:")
    print(f"Sample Input (Flattened): \n{sample_input.ravel()}")
    print(f"Actual Label (One-Hot Encoded): \n{sample_label}")
    print(f"Untrained Model Output: \n{untrained_output}")

    # Train the model
    model.train_model(train_images, train_labels)

    # Get model output for the trained model
    trained_output = model.process_raw_single_input(sample_input)
    print("\nAfter Training:")
    print(f"Sample Input (Flattened): \n{sample_input.ravel()}")
    print(f"Actual Label (One-Hot Encoded): \n{sample_label}")
    print(f"Trained Model Output: \n{trained_output}")




if __name__ == "__main__":
    main()