import numpy as np
import matplotlib.pyplot as plt
from time import time
from pso import PSO

# Ackley function
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# Levy function
def levy(x):
    d = len(x)
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    sum_terms = np.sum([(w[i] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2) for i in range(d - 1)])
    return term1 + sum_terms + term3

# Fourier series MSE for Square Wave Approximation
def square_wave_fourier_mse(coeffs, t=np.linspace(0, 2 * np.pi, 100)):
    square_wave = np.sign(np.sin(t))
    approx_wave = np.zeros_like(t)
    num_coeffs = len(coeffs) // 2
    a_coeffs = coeffs[:num_coeffs]
    b_coeffs = coeffs[num_coeffs:]
    for n in range(1, num_coeffs + 1):
        approx_wave += a_coeffs[n - 1] * np.cos(n * t) + b_coeffs[n - 1] * np.sin(n * t)
    mse = np.mean((square_wave - approx_wave)**2)
    return mse

# PSO optimization function for given functions
def run_pso(func, dimensions=2, learning_rate=0.5, power=10, iterations=100):
    pso = PSO(func, num_particles=30, dimensions=dimensions, w=learning_rate)
    start_time = time()
    best_position = pso.optimize(iterations=iterations)
    duration = time() - start_time
    mse_over_epochs = pso.global_best_scores
    final_mse = func(best_position)
    mse_variation = np.std(mse_over_epochs)
    return best_position, final_mse, mse_variation, duration, mse_over_epochs

# Parameters and results storage
learning_rates = [0.1, 0.3, 0.5, 0.7]
powers = [5, 10, 15, 20]
epochs = 100
results = {"ackley": [], "levy": [], "fourier": []}

# Run experiments and store results
for lr in learning_rates:
    for power in powers:
        # Ackley function
        best_position, final_mse, mse_variation, duration, mse_over_epochs = run_pso(
            ackley, dimensions=2, learning_rate=lr, iterations=epochs
        )
        results["ackley"].append({
            "learning_rate": lr, "power": power, "final_mse": final_mse,
            "mse_variation": mse_variation, "time": duration, "mse_over_epochs": mse_over_epochs
        })

        # Levy function
        best_position, final_mse, mse_variation, duration, mse_over_epochs = run_pso(
            levy, dimensions=2, learning_rate=lr, iterations=epochs
        )
        results["levy"].append({
            "learning_rate": lr, "power": power, "final_mse": final_mse,
            "mse_variation": mse_variation, "time": duration, "mse_over_epochs": mse_over_epochs
        })

        # Fourier series approximation
        best_coeffs, final_mse, mse_variation, duration, mse_over_epochs = run_pso(
            lambda c: square_wave_fourier_mse(c, t=np.linspace(0, 2 * np.pi, 100)),
            dimensions=2 * power, learning_rate=lr, power=power, iterations=epochs
        )
        results["fourier"].append({
            "learning_rate": lr, "power": power, "final_mse": final_mse,
            "mse_variation": mse_variation, "time": duration, "mse_over_epochs": mse_over_epochs,
            "best_coeffs": best_coeffs
        })

# Plotting for (a) - (d) for each function
for key, func_results in results.items():
    learning_rate_values = [r["learning_rate"] for r in func_results]
    power_values = [r["power"] for r in func_results]
    mse_values = [r["final_mse"] for r in func_results]
    mse_variation_values = [r["mse_variation"] for r in func_results]
    mse_epoch_curves = [r["mse_over_epochs"] for r in func_results]

    # (a) Learning Rate vs MSE
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(learning_rate_values, mse_values, color="b")
    plt.title(f"Learning Rate vs. Final MSE ({key.capitalize()})")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final MSE")

    # (b) Learning Rate vs. MSE Variation
    plt.subplot(2, 2, 2)
    plt.scatter(learning_rate_values, mse_variation_values, color="g")
    plt.title(f"Learning Rate vs. MSE Variation ({key.capitalize()})")
    plt.xlabel("Learning Rate")
    plt.ylabel("MSE Variation")

    # (c) Power vs MSE
    plt.subplot(2, 2, 3)
    plt.scatter(power_values, mse_values, color="r")
    plt.title(f"Power vs. Final MSE ({key.capitalize()})")
    plt.xlabel("Power")
    plt.ylabel("Final MSE")

    # (d) Epoch vs MSE for each combination
    plt.subplot(2, 2, 4)
    for i, mse_over_epochs in enumerate(mse_epoch_curves):
        plt.plot(range(epochs), mse_over_epochs, label=f"LR={learning_rate_values[i]}, Power={power_values[i]}")
    plt.title(f"Epoch vs. MSE ({key.capitalize()})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
    plt.tight_layout()
    plt.show()

# Best Fourier series model plot
best_fourier_result = min(results["fourier"], key=lambda x: x["final_mse"])
t = np.linspace(0, 2 * np.pi, 100)
ideal_square_wave = np.sign(np.sin(t))
approx_square_wave = np.zeros_like(t)
num_coeffs = len(best_fourier_result["best_coeffs"]) // 2
a_coeffs = best_fourier_result["best_coeffs"][:num_coeffs]
b_coeffs = best_fourier_result["best_coeffs"][num_coeffs:]
for n in range(1, num_coeffs + 1):
    approx_square_wave += a_coeffs[n - 1] * np.cos(n * t) + b_coeffs[n - 1] * np.sin(n * t)

plt.figure()
plt.plot(t, ideal_square_wave, label="Ideal Square Wave", linestyle='--')
plt.plot(t, approx_square_wave, label="Fourier Approximation")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Square Wave Approximation using Optimized Fourier Coefficients (Best Model)")
plt.legend()
plt.show()
