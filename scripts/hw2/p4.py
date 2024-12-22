import numpy as np
import matplotlib.pyplot as plt
from pso import PSO

# First-order plus time-delay system parameters
time_delay = 1.0  # Time delay (τDT)
time_constant = 10.0  # Time constant (τ)
K = 1.0  # System gain

# PID controller simulation function
def pid_controller(Kp, Ki, Kd, setpoint, dt=0.1, time_end=20):
    num_steps = int(time_end / dt)
    t = np.linspace(0, time_end, num_steps)
    output = np.zeros(num_steps)
    error_sum = 0  # Integral term
    previous_error = 0  # For derivative term
    for i in range(1, num_steps):
        time_delay_index = max(0, i - int(time_delay / dt))
        process_value = output[time_delay_index]  # Delayed output
        error = setpoint - process_value
        error_sum += error * dt
        d_error = (error - previous_error) / dt
        previous_error = error

        # PID control law
        control_signal = Kp * error + Ki * error_sum + Kd * d_error

        # System response (First-order system)
        output[i] = output[i-1] + (dt / time_constant) * (K * control_signal - output[i-1])
    return t, output

# Performance metric (Sum of Square Error - SSE)
def performance_metric(params):
    Kp, Ki, Kd = params
    _, response = pid_controller(Kp, Ki, Kd, setpoint=1.0)
    error = 1.0 - response  # Error against setpoint
    sse = np.sum(error**2)
    return sse

# Run PSO for PID optimization using the provided PSO class
pso_pid = PSO(performance_metric, num_particles=30, dimensions=3, w=0.5, c1=2, c2=2)
best_pid_params = pso_pid.optimize(iterations=100)
print("Optimal PID parameters (Kp, Ki, Kd):", best_pid_params)

# Plot optimization curve for SSE
pso_pid.plot_best_scores()

# Plot system response with optimized PID parameters
time, optimized_response = pid_controller(*best_pid_params, setpoint=1.0)
plt.plot(time, optimized_response, label="Optimized PID Response")
plt.axhline(1.0, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("System Output")
plt.title("System Response with Optimized PID Parameters")
plt.legend()
plt.show()
