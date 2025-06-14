from sklearn.model_selection import train_test_split
import os
import random

def load_dataset(dataset_path, batch_size):
    """
    Function to load and preprocess the dataset.
    Combines data from ONE (label 1) and ZERO (label 0),
    shuffles them, and splits into 70% training and 30% testing.
    Returns batches of data according to batch_size.
    """
    print("Loading dataset...")

    # Paths to the datasets
    one_path = os.path.join(dataset_path, "ONE")
    zero_path = os.path.join(dataset_path, "ZERO")

    # Collect all file paths with labels
    data_files = []

    # Add files from ONE with label 1
    for root, dirs, files in os.walk(one_path):
        for file in files:
            if file.endswith(".mp4"):  # Adjust the extension if needed
                data_files.append((os.path.join(root, file), 1))

    # Add files from ZERO with label 0
    for root, dirs, files in os.walk(zero_path):
        for file in files:
            if file.endswith(".mp4"):  # Adjust the extension if needed
                data_files.append((os.path.join(root, file), 0))

    # Shuffle the data
    random.shuffle(data_files)

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data_files, test_size=0.3, random_state=42)

    print(f"Total files: {len(data_files)}, Train: {len(train_data)}, Test: {len(test_data)}")

    # Helper function to create batches
    def create_batches(data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    train_batches = create_batches(train_data, batch_size)
    test_batches = create_batches(test_data, batch_size)

    return train_batches, test_batches