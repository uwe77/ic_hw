import numpy as np
import matplotlib.pyplot as plt
from pso import PSO


# Define Levy function for optimization
def levy(x):
    d = len(x)
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    sum_terms = np.sum([(w[i] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[i] + 1))**2) for i in range(d - 1)])
    return term1 + sum_terms + term3

# Initialize PSO for Levy function optimization
pso_levy = PSO(levy, num_particles=30, dimensions=2)
best_levy_position = pso_levy.optimize(iterations=100)
print("Optimal position for Levy function:", best_levy_position)
print("Optimal Levy function value:", levy(best_levy_position))
pso_levy.plot_best_scores()

# Fourier Series for Square Wave Approximation - Mean Square Error Function
def square_wave_fourier_mse(coeffs, t=np.linspace(0, 2 * np.pi, 100)):
    # Define the target square wave (ideal)
    square_wave = np.sign(np.sin(t))
    
    # Fourier series approximation
    approx_wave = np.zeros_like(t)
    num_coeffs = len(coeffs) // 2
    a_coeffs = coeffs[:num_coeffs]
    b_coeffs = coeffs[num_coeffs:]
    
    for n in range(1, num_coeffs + 1):
        approx_wave += a_coeffs[n - 1] * np.cos(n * t) + b_coeffs[n - 1] * np.sin(n * t)
    
    # Calculate MSE between the ideal and the approximation
    mse = np.mean((square_wave - approx_wave)**2)
    return mse

# Initialize PSO for Fourier coefficients optimization
num_coeffs = 10  # number of coefficients to optimize (5 a_n and 5 b_n)
pso_fourier = PSO(square_wave_fourier_mse, num_particles=30, dimensions=2 * num_coeffs)
best_fourier_coeffs = pso_fourier.optimize(iterations=100)
print("Optimal Fourier coefficients for square wave approximation:", best_fourier_coeffs)
print("Optimal MSE for square wave approximation:", square_wave_fourier_mse(best_fourier_coeffs))
pso_fourier.plot_best_scores()

# Plot the square wave approximation using optimized Fourier coefficients
t = np.linspace(0, 2 * np.pi, 100)
ideal_square_wave = np.sign(np.sin(t))
approx_square_wave = np.zeros_like(t)

# Apply the optimized coefficients
a_coeffs = best_fourier_coeffs[:num_coeffs]
b_coeffs = best_fourier_coeffs[num_coeffs:]
for n in range(1, num_coeffs + 1):
    approx_square_wave += a_coeffs[n - 1] * np.cos(n * t) + b_coeffs[n - 1] * np.sin(n * t)

# Plotting the results
plt.plot(t, ideal_square_wave, label="Ideal Square Wave", linestyle='--')
plt.plot(t, approx_square_wave, label="Fourier Approximation")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Square Wave Approximation using Optimized Fourier Coefficients")
plt.legend()
plt.show()
