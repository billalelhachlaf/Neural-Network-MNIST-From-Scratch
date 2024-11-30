import numpy as np
import pandas as pd


training_data = pd.read_csv('train.csv')
training_labels_before = training_data['label'].to_numpy()  # This will be a (N,) array where N is the number of samples
training_pixels = training_data.drop(columns=['label']).to_numpy()  # This will be a (N, 784) array
training_pixels = training_pixels / 255.0  # Normalize pixel values

num_classes = 10
training_labels = np.eye(num_classes)[training_labels_before]  # Convert labels to one-hot encoding




test_data = pd.read_csv('test.csv')
test_pixels = test_data.to_numpy()
test_pixels = test_pixels / 255.0
