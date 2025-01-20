import numpy as np
import gzip
import os
import urllib.request

# Download and Load Fashion MNIST Dataset
base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
dataset_dir = "fashion_mnist_data"
os.makedirs(dataset_dir, exist_ok=True)

def download_file(filename, url):
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url + filename, filepath)
        print(f"{filename} downloaded.")
    return filepath

def parse_idx(filepath):
    with gzip.open(filepath, "rb") as f:
        data = f.read()
    magic_number = int.from_bytes(data[:4], "big")
    num_items = int.from_bytes(data[4:8], "big")
    if magic_number == 2051:  # Images
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_items, rows * cols)
        return images
    elif magic_number == 2049:  # Labels
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        return labels

# Download and parse data
train_images_path = download_file(files["train_images"], base_url)
train_labels_path = download_file(files["train_labels"], base_url)
test_images_path = download_file(files["test_images"], base_url)
test_labels_path = download_file(files["test_labels"], base_url)

x_train = parse_idx(train_images_path) / 255.0  # Normalize images
y_train = parse_idx(train_labels_path)
x_test = parse_idx(test_images_path) / 255.0
y_test = parse_idx(test_labels_path)

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Neural Network Implementation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return z > 0

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            # Shuffle data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)
            
            # Compute loss for the entire dataset
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Hyperparameters
input_size = 784  # 28x28 images flattened
hidden_size = 128
output_size = 10  # 10 classes
learning_rate = 0.01
epochs = 50
batch_size = 64

# Initialize and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(x_train, y_train, epochs, batch_size)

# Evaluate on the test set
y_test_pred = nn.forward(x_test)
test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {test_accuracy:.4f}")
