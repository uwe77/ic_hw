from mlp import MLP
import numpy as np


# XOR input and output
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Train XOR using MLP
xor_mlp = MLP(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)
xor_mlp.train(X_xor, y_xor, epochs=5000)

# Test XOR model and print results
for x in X_xor:
    output = xor_mlp.forward(x)
    print(f"Input: {x}, Predicted: {output.round()}, Actual: {output}")

# Plot training loss curve
xor_mlp.plot_loss()