from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Define the function mappings
def func1(x):
    return 1 / x

def func2(x, y):
    return x**2 + 3 * y

def func3(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# Function to generate datasets
def generate_data(func, input_range, n_samples=200):
    if func == func1:
        x = np.random.uniform(input_range[0], input_range[1], (n_samples, 1))
        y = func(x)
        return x, y
    else:
        x = np.random.uniform(input_range[0], input_range[1], (n_samples, 1))
        y = np.random.uniform(input_range[0], input_range[1], (n_samples, 1))
        inputs = np.hstack((x, y))
        outputs = func(x, y)
        return inputs, outputs

# Train MLP on each function and plot results
def train_and_plot(func, input_size, output_size, input_range, epochs=10000, net_arch=np.array([10]), act_func="relu", learning_rate=0.001):
    # Generate data
    X, y = generate_data(func, input_range, n_samples=200)
    X_test, y_test = generate_data(func, input_range, n_samples=40)
    
    # Normalize data (optional but recommended)
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
    X = (X - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y = (y - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Define network architecture and activation functions
    net_arch = np.insert(net_arch, 0, input_size)
    net_arch = np.append(net_arch, output_size)
    print(f"Network Architecture: {net_arch}")
    activations = []
    for i in range(len(net_arch) - 1):
        if i == len(net_arch) - 2:
            activations.append("linear")
        else:
            activations.append(act_func)
    print(f"Activations: {activations}")
    
    # Initialize MLP with flexible architecture
    mlp = MLP(net_arch=net_arch, activations=activations, learning_rate=learning_rate)
    
    # Train the MLP and plot the loss
    mlp.train(X, y, epochs=epochs)
    # mlp.save_model(f"mlp_{func.__name__}.pkl")
    # mlp.load_model(f"mlp_{func.__name__}.pkl")
    mlp.plot_loss()

    # Test on the test data
    predictions = mlp.forward(X_test)
    # Denormalize predictions and y_test
    predictions = predictions * y_std + y_mean
    y_test_denorm = y_test * y_std + y_mean
    
    # Calculate error
    err = np.mean((predictions - y_test_denorm) ** 2)
    err_percent = 100 * err / np.mean(y_test_denorm ** 2)
    print(f"Mean Squared Error on Test Data: {err:.4f}")
    print(f"Mean Squared Error (% of mean y^2): {err_percent:.2f}%")
    
    # Sort test data for better plotting
    if input_size == 1:
        sorted_indices = np.argsort(X_test[:, 0], axis=0).flatten()
        X_test_sorted = X_test[sorted_indices]
        plt.plot(X_test_sorted * X_std + X_mean, predictions[sorted_indices], label=f"{func.__name__} Predictions")
        plt.plot(X_test_sorted * X_std + X_mean, y_test_denorm[sorted_indices], label="Actual")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_test[:, 0] * X_std[0] + X_mean[0], X_test[:, 1] * X_std[1] + X_mean[1], predictions, label=f"{func.__name__} Predictions")
        ax.scatter(X_test[:, 0] * X_std[0] + X_mean[0], X_test[:, 1] * X_std[1] + X_mean[1], y_test_denorm, label="Actual")
    
    plt.title(f"MLP Approximation for {func.__name__}")
    plt.legend()
    plt.show()

# Training MLP on each function
if __name__ == "__main__":
    # Function 1: f(x) = 1 / x, x ∈ [0.1, 1]
    train_and_plot(func1, input_size=1, output_size=1, input_range=(0.1, 1), epochs=10000,
                   net_arch=np.array([20]),
                   act_func="tanh",
                   learning_rate=0.001)
    
    # Function 2: f(x, y) = x^2 + 3y, x ∈ [-1, 1], y ∈ [-1, 1]
    train_and_plot(func2, input_size=2, output_size=1, input_range=(-1, 1), epochs=10000,
                   net_arch=np.array([10]),
                   act_func="relu",
                   learning_rate=0.001)
    
    # Function 3: f(x, y) = sin(πx) cos(πy), x ∈ [-2, 2], y ∈ [-2, 2]
    train_and_plot(func3, input_size=2, output_size=1, input_range=(-2, 2), epochs=10000, 
                   net_arch=np.array([5]),
                   act_func="relu",
                   learning_rate=0.001)
