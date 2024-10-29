from mlp import MLP
import numpy as np


# Generate and train the MLP on different mappings
X = np.random.uniform(0.1, 1, (200, 1))
y = 1 / X
mlp = MLP(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
mlp.train(X, y)
mlp.plot_loss()