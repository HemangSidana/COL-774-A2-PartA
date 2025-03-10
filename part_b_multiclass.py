import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import argparse
# from numpy import numpy_transforms    #TODO: check
import pickle

#Remember to import "numpy_transforms" functions if you wish to import these two classes in a different script.

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_root', type=str, required=True, help='Directory with all the subfolders.')
parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')

args = parser.parse_args()

root_dir = args.dataset_root
weights_saving_path = args.save_weights_path

np.random.seed(0)

class CustomImageDataset:
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L") #Convert image to greyscale
        label = row["class"]

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

# Transformations using NumPy
def resize(image, size):
    # return np.array(Image.fromarray(image).resize(size))
    return np.array(image.resize(size))

def to_tensor(image):
    return image.astype(np.float32) / 255.0

def numpy_transform(image, size=(25, 25)):
    image = resize(image, size)
    image = to_tensor(image)
    image = image.flatten()
    return image

class DataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        # if self.shuffle:
        #     np.random.shuffle(self.indices)

    def __iter__(self):
        self.start_idx = 0
        return self
    def __len__(self):
        return int(len(self.dataset)/self.batch_size)

    def __next__(self):
        if self.start_idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.start_idx:end_idx]
        images = []
        labels = []

        for idx in batch_indices:
            image, label = self.dataset[idx]
            images.append(image)
            labels.append(label)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)
        batch_labels = np.array(labels)

        return batch_images, batch_labels
    


# Root directory containing the 8 subfolders
mode = 'train' #Set mode to 'train' for loading the train set for training. Set mode to 'val' for testing your model after training. 

if mode == 'train': # Set mode to train when using the dataloader for training the model.
    csv = os.path.join(root_dir, "train.csv")

elif mode == 'val':
    csv = os.path.join(root_dir, "val.csv")

# Create the custom dataset
dataset = CustomImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=256)

def one_hot_encode(y, num_classes):
    # Convert y to a 2D one-hot encoding matrix
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1
    return y_one_hot

batches=[]
for images,labels in dataloader:
    one_hot_labels= one_hot_encode(labels,8)
    batches.append((images,one_hot_labels))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x, axis=None):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Neural Network Class with Softmax in the Output Layer and Sigmoid in Hidden Layers
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(np.float64) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1]), dtype=np.float64))

    def forward(self, X):
        activations = [X]
        pre_activations = []

        # Pass through each layer except the output layer
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = sigmoid(z)  # Sigmoid for hidden layers
            activations.append(a)

        # Pass through the output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z)
        a = softmax(z, axis=1)  # Softmax for the output layer
        activations.append(a)

        return activations, pre_activations

    def backward(self, X, y, activations, pre_activations):
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # Start with output layer error
        delta = activations[-1] - y

        for i in reversed(range(len(self.weights))):
            grad_w[i] = np.dot(activations[i].T, delta) / delta.shape[0]
            grad_b[i] = np.sum(delta, axis=0, keepdims=True) / delta.shape[0]

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(pre_activations[i - 1])


        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]

    def train(self, batches, epochs, learning_rate):
        for epoch in range(epochs):
            for X_batch, y_batch in batches:
                activations, pre_activations = self.forward(X_batch)
                grad_w, grad_b = self.backward(X_batch, y_batch, activations, pre_activations)
                self.update_parameters(grad_w, grad_b, learning_rate)

            # Calculate average loss over batches
            loss = 0
            for X_batch, y_batch in batches:
                y_pred, _ = self.forward(X_batch)
                loss += cross_entropy_loss(y_batch, y_pred[-1])

            loss /= len(batches)
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.10f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

# Example usage:
nn = NeuralNetwork(625, [512, 256, 128], 8)
nn.train(batches, 15, 0.001)

# Number of layers in the Neural Network
N = 4  # Example value, replace with the actual number of layers

# Initialize the dictionary
weights_dict = {
    'weights': {},
    'bias': {}
}

weights = nn.get_weights()
biases = nn.get_biases()

# Populate the weights and bias dictionaries
for i in range(N):
    weights_dict['weights'][f'fc{i+1}'] = weights[i]
    weights_dict['bias'][f'fc{i+1}'] = biases[i].flatten()

# Save the dictionary as a pickle file
with open(weights_saving_path, 'wb') as f:
    pickle.dump(weights_dict, f)