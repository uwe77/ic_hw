import numpy as np
import matplotlib.pyplot as plt

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# MLP class definition
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        self.losses = []
    
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, x, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)
        
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.weights_input_hidden += x.T.dot(d_hidden) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean((y - output) ** 2)
            self.losses.append(loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.show()