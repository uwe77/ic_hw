import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Dictionary to map activation names to functions
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative)  # Added linear activation
}


# MLP class definition
class MLP:
    def __init__(self, net_arch, activations, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.net_arch = net_arch
        self.activations = [activation_functions[act] for act in activations]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(net_arch) - 1):
            self.weights.append(np.random.rand(net_arch[i], net_arch[i + 1]))
            self.biases.append(np.random.rand(net_arch[i + 1]))

        self.losses = []

    def forward(self, x):
        self.a = [x]  # store all layer outputs for backward pass
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            activation_func = self.activations[i][0]
            self.a.append(activation_func(z))
        return self.a[-1]

    def backward(self, y):
        deltas = [None] * len(self.weights)  # to store delta values
        error = y - self.a[-1]
        delta = error * self.activations[-1][1](self.a[-1])
        deltas[-1] = delta

        # Backpropagate through layers
        for i in reversed(range(len(deltas) - 1)):
            error = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = error * self.activations[i][1](self.a[i + 1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += self.a[i].T.dot(deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y)
            loss = np.mean((y - output) ** 2)
            self.losses.append(loss)
            # make the print stay at the same line
            sys.stdout.write(f"\rEpoch: {epoch}, Loss: {loss:.6f}")
            sys.stdout.flush()
        print("\nTraining complete!")

    def save_model(self, file_path):
        """
        Save the model parameters to a file using pickle.

        Parameters:
        - file_path: Path to the file where the model will be saved.
        """
        model_data = {
            'net_arch': self.net_arch,
            'activations': self.activations[0],  # Save activation names
            'weights': self.weights,
            'biases': self.biases,
            'learning_rate': self.learning_rate,
            'losses': self.losses
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load the model parameters from a file using pickle.

        Parameters:
        - file_path: Path to the file from which the model will be loaded.

        Returns:
        - An instance of MLP with loaded parameters.
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct the MLP instance using activation names
        model = MLP(
            net_arch=model_data['net_arch'],
            activations=model_data['activations'],
            learning_rate=model_data['learning_rate']
        )
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        model.losses = model_data.get('losses', [])
        print(f"Model loaded from {file_path}")
        return model

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.show()
