from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt

# Define XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Initialize MLP for XOR problem
net_arch = np.array([2, 10, 1])
xor_mlp = MLP(net_arch=net_arch, activations="relu", learning_rate=0.001)

# Train the MLP on XOR data
xor_mlp.train(X_xor, y_xor, epochs=100000)
# Plot the training loss to observe convergence
xor_mlp.plot_loss()

# Test the MLP on the XOR inputs and print results
print("Testing XOR MLP results:")
for x, y in zip(X_xor, y_xor):
    output = xor_mlp.forward(x)
    print(f"Input: {x}, Predicted: {output.round()}, Actual: {y}")
