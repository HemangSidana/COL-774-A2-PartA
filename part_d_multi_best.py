import argparse

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import pickle


# Set up argument parser
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_root', type=str, required=True, help='Directory with all the subfolders.')
parser.add_argument('--test_dataset_root', type=str, required=True, help='Directory with the test dataset.')
parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save the weights.')
parser.add_argument('--save_predictions_path', type=str, required=True, help='Path to save the predictions.')

args = parser.parse_args()

root_dir = args.dataset_root
test_root_dir = args.test_dataset_root
weights_saving_path = args.save_weights_path
predictions_saving_path = args.save_predictions_path

#Remember to import "numpy_transforms" functions if you wish to import these two classes in a different script.

np.random.seed(0)

class TrainImageDataset:
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

class TrainDataLoader:
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


csv = os.path.join(root_dir, "train.csv") #The csv file will always be placed inside the dataset directory.
#Please ensure that you set the csv file path accordingly in all parts of the assignment so that it gets loaded correctly.
#While evaluation, the train.csv will have same name while the test set csv will be renamed to "val.csv" to be compatible with the setting here.

# Create the custom dataset
dataset = TrainImageDataset(root_dir=root_dir, csv = csv, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.

# Create the DataLoader
dataloader = TrainDataLoader(dataset, batch_size=256)

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

# Leaky ReLU activation and its derivative
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x, axis=None):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Avoid log(0)
    return -np.sum(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# Neural Network Class with Leaky ReLU and Adam Optimizer
class NeuralNetwork_Adam:
    def __init__(self, input_size, hidden_sizes, output_size, init_weights=None, init_biases=None, init_seed=None, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # if init_seed is None:
        #     self.best_seed = int(time.time())
        #     np.random.seed(self.best_seed)
        # else:
        #     np.random.seed(init_seed)
        np.random.seed(0)
        self.weights = []
        self.biases = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for Adam
        self.alpha = alpha  # Leaky ReLU parameter
        self.best_weights = []
        self.best_biases = []
        self.best_loss = float("inf")

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights, biases, and Adam parameters (m, v)
        for i in range(len(layer_sizes) - 1):
            if init_weights is not None and init_biases is not None:
                self.weights.append(init_weights[i])
                self.biases.append(init_biases[i])
            else:
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]).astype(np.float64) * np.sqrt(2 / layer_sizes[i]))
                self.biases.append(np.zeros((1, layer_sizes[i + 1]), dtype=np.float64))
            self.m_w.append(np.zeros_like(self.weights[-1]))
            self.v_w.append(np.zeros_like(self.weights[-1]))
            self.m_b.append(np.zeros_like(self.biases[-1]))
            self.v_b.append(np.zeros_like(self.biases[-1]))
        # self.best_weights = self.weights
        # self.best_biases = self.biases

    def forward(self, X):
        activations = [X]
        pre_activations = []

        # Pass through each hidden layer
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            a = leaky_relu(z, alpha=self.alpha)  # Leaky ReLU for hidden layers
            # a = sigmoid(z)
            activations.append(a)

        # Pass through the output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        pre_activations.append(z)
        a = softmax(z, axis=1)  # Softmax for the output layer
        activations.append(a)

        return activations, pre_activations

    def forward_pred(self, X):
        activations = [X]
        pre_activations = []

        # Pass through each hidden layer
        for i in range(len(self.best_weights) - 1):
            z = np.dot(activations[-1], self.best_weights[i]) + self.best_biases[i]
            pre_activations.append(z)
            a = leaky_relu(z, alpha=self.alpha)  # Leaky ReLU for hidden layers
            # a = sigmoid(z)
            activations.append(a)

        # Pass through the output layer with softmax
        z = np.dot(activations[-1], self.best_weights[-1]) + self.best_biases[-1]
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
                delta = np.dot(delta, self.weights[i].T) * leaky_relu_derivative(pre_activations[i - 1], alpha=self.alpha)
                # delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(pre_activations[i - 1])

        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b, learning_rate):
        self.t += 1  # Increment time step for Adam

        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]

            # Update biased second moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)

            # Compute bias-corrected first moment estimate
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment estimate
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update weights and biases
            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def train(self, batches, time_of_running, learning_rate):
        start_time = time.time()
        epoch = 0
        while epoch < 2250:
            for X_batch, y_batch in batches:
                activations, pre_activations = self.forward(X_batch)
                grad_w, grad_b = self.backward(X_batch, y_batch, activations, pre_activations)
                self.update_parameters(grad_w, grad_b, learning_rate)

            # Calculate average loss over batches
            loss = 0
            z = 0
            for X_batch, y_batch in batches:
                y_pred, _ = self.forward(X_batch)
                loss += cross_entropy_loss(y_batch, y_pred[-1])
                z += len(y_pred[-1])
            loss /= z
            
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_weights = [np.copy(w) for w in self.weights]  # Use np.copy to create independent copies
                self.best_biases = [np.copy(b) for b in self.biases]  # Use np.copy for biases
            # print(f"Epoch {epoch + 1}, Loss: {loss:.10f}")
            # get_stat()
            epoch += 1
            # if time elapsed is greater than 1 minute, break the loop
            if time.time() - start_time > 60 * time_of_running:
                break

    def predict(self, X):
        activations, _ = self.forward_pred(X)
        return activations[-1]
    
    def get_best_weights(self):
        return self.best_weights
    
    def get_best_biases(self):
        return self.best_biases
    
    def get_best_loss(self):
        return self.best_loss
    
    # def get_best_seed(self):
    #     return self.best_seed


best_loss = float('inf')
best_weights_init = []
best_biases_init = []
best_weights = []
best_biases = []
# best_seed = 0

for _ in range (1):
    nn = NeuralNetwork_Adam(625, [256,128,64], 8, beta1=0.95, beta2=0.999)
    nn.train(batches,14,learning_rate=0.0015)
    if nn.get_best_loss() < best_loss:
        best_loss = nn.get_best_loss()
        best_weights = nn.get_best_weights()
        best_biases = nn.get_best_biases()
        # best_seed = nn.get_best_seed()

# print(best_loss)

# Number of layers in the Neural Network
N = 4  # Example value, replace with the actual number of layers

# Initialize the dictionary
weights_dict = {
    'weights': {},
    'bias': {}
}

weights = best_weights
biases = best_biases

# Populate the weights and bias dictionaries
for i in range(N):
    weights_dict['weights'][f'fc{i+1}'] = weights[i]
    weights_dict['bias'][f'fc{i+1}'] = biases[i].flatten()

# Save the dictionary as a pickle file
with open(weights_saving_path, 'wb') as f:
    pickle.dump(weights_dict, f)

np.random.seed(0)

class TestImageDataset:
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

        if self.transform:
            image = self.transform(image)

        return np.array(image)

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

class TestDataLoader:
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
            image = self.dataset[idx]
            images.append(image)

        self.start_idx = end_idx

        # Stack images and labels to create batch tensors
        batch_images = np.stack(images, axis=0)

        return batch_images
    
# Root directory containing the 8 subfolders
csv = os.path.join(root_dir, "val.csv") #The csv file will always be placed inside the dataset directory.
#Please ensure that you set the csv file path accordingly in all parts of the assignment so that it gets loaded correctly.
#While evaluation, the train.csv will have same name while the test set csv will be renamed to "val.csv" to be compatible with the setting here.

# Create the custom dataset
dataset = TestImageDataset(root_dir=test_root_dir, csv = csv, transform=numpy_transform)  #Remember to import "numpy_transforms" functions.

# Create the DataLoader
dataloader = TestDataLoader(dataset, batch_size=1)

# def one_hot_encode(y, num_classes):
#     # Convert y to a 2D one-hot encoding matrix
#     y_one_hot = np.zeros((len(y), num_classes))
#     y_one_hot[np.arange(len(y)), y] = 1
#     return y_one_hot

batches=[]
for images in dataloader:
    # one_hot_labels= one_hot_encode(labels,8)
    batches.append(images)

predictions = []
for X_val in batches:
    Y_pred= nn.predict(X_val)
    predictions.append(Y_pred)
    # print(cross_entropy_loss(Y_val,Y_pred)/len(dataset))
    # print(accuracy(Y_val, Y_pred))

predictions = np.array(predictions)
predictions = np.argmax(predictions, axis=2)
predictions = predictions.flatten()

# print(Y_pred.shape)
# print(Y_pred[:10])

# save y_pred in pickle file
with open(predictions_saving_path, 'wb') as f:
    pickle.dump(predictions, f)