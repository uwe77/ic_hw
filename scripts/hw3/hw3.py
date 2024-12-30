import numpy as np
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
from skfuzzy import control as ctrl
from scipy.signal import lti, step
from matplotlib import pyplot as plt
from pso import PSO  # Assuming PSO is provided and implemented

# --- Problem 1(a): Fuzzy Controller Design ---

# Define fuzzy variables
error = Antecedent(np.linspace(-1, 1, 100), 'error')
delta_error = Antecedent(np.linspace(-0.5, 0.5, 100), 'delta_error')
integral_error = Antecedent(np.linspace(0, 10, 100), 'integral_error')
control_output = Consequent(np.linspace(-1, 1, 100), 'control_output')

# Membership functions
error['negative'] = fuzz.trapmf(error.universe, [-1, -1, -0.5, 0])
error['zero'] = fuzz.trimf(error.universe, [-0.5, 0, 0.5])
error['positive'] = fuzz.trapmf(error.universe, [0, 0.5, 1, 1])

delta_error['negative'] = fuzz.trapmf(delta_error.universe, [-0.5, -0.5, -0.25, 0])
delta_error['zero'] = fuzz.trimf(delta_error.universe, [-0.25, 0, 0.25])
delta_error['positive'] = fuzz.trapmf(delta_error.universe, [0, 0.25, 0.5, 0.5])

integral_error['low'] = fuzz.trimf(integral_error.universe, [0, 2.5, 5])
integral_error['medium'] = fuzz.trimf(integral_error.universe, [2.5, 5, 7.5])
integral_error['high'] = fuzz.trimf(integral_error.universe, [5, 7.5, 10])

control_output['negative'] = fuzz.trapmf(control_output.universe, [-1, -1, -0.5, 0])
control_output['zero'] = fuzz.trimf(control_output.universe, [-0.5, 0, 0.5])
control_output['positive'] = fuzz.trapmf(control_output.universe, [0, 0.5, 1, 1])

# Define rules
rule1 = Rule(error['positive'] & delta_error['zero'], control_output['positive'])
rule2 = Rule(error['negative'] & delta_error['negative'], control_output['negative'])
rule3 = Rule(error['zero'] & integral_error['medium'], control_output['zero'])
rule4 = Rule(error['positive'] & integral_error['high'], control_output['negative'])
rule5 = Rule(error['negative'] & integral_error['low'], control_output['positive'])

# Create the control system
control_system = ControlSystem([rule1, rule2, rule3, rule4, rule5])
sim = ControlSystemSimulation(control_system)

# Test the controller
errors = np.linspace(-1, 1, 100)
responses = []

for e in errors:
    sim.input['error'] = e
    sim.input['delta_error'] = 0
    sim.input['integral_error'] = 5
    sim.compute()
    responses.append(sim.output['control_output'])

# Plot fuzzy controller response
plt.plot(errors, responses, label="Fuzzy Controller Output")
plt.title("Fuzzy Controller Response (PID-like)")
plt.xlabel("Error")
plt.ylabel("Control Output")
plt.grid()
plt.legend()
plt.show()

# --- Problem 1(b): PSO for PID Optimization ---

# Plant Transfer Function
num = [1]
den = [10, 1]
plant = lti(num, den)

def performance_index(params):
    kp, kd, ki = params
    num_c = [kd, kp, ki]
    den_c = [1, 0]
    controller = lti(num_c, den_c)

    # Closed-loop transfer function
    system = lti(
        np.convolve(plant.num, controller.num),
        np.polyadd(np.convolve(plant.den, controller.den), np.convolve(plant.num, controller.num)),
    )

    # Simulate step response
    t = np.linspace(0, 50, 500)
    t_out, y = step(system, T=t)

    # Performance metrics
    rise_time = np.argmax(y >= 0.9 * max(y)) * (t[1] - t[0])
    overshoot = max(y) - 1

    settling_indices = np.where(np.abs(y - 1) < 0.02)[0]
    if settling_indices.size > 0:
        settling_time = t[settling_indices[-1]]
    else:
        settling_time = t[-1]

    sse = np.sum((y - 1) ** 2)

    return 0.3 * rise_time + 0.3 * settling_time + 0.2 * overshoot + 0.2 * sse

# Initialize PSO
pso = PSO(performance_index, num_particles=30, dimensions=3, w=0.5, c1=2, c2=2)
best_params = pso.optimize(iterations=100)
kp, kd, ki = best_params

# Simulate optimized response
num_c = [kd, kp, ki]
den_c = [1, 0]
controller = lti(num_c, den_c)
system = lti(
    np.convolve(plant.num, controller.num),
    np.polyadd(np.convolve(plant.den, controller.den), np.convolve(plant.num, controller.num)),
)

# Plot Step Response
t = np.linspace(0, 50, 500)
t_out, y = step(system, T=t)

plt.figure()
plt.plot(t_out, y, label="Optimized PID Response")
plt.title("Optimized Step Response (PID)")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid()
plt.legend()
plt.show()

# Plot Optimization Curve
pso.plot_best_scores()

print("Optimized PID parameters:")
print(f"Kp: {kp}, Kd: {kd}, Ki: {ki}")

# --- Problem 2: Fuzzy Cruise Control Simulation ---

# System Parameters
mass = 1500  # kg
Tm = 190      # Nm
omega_n = 420 # rad/s
alpha_n = 25  # Gear ratio
beta = 0.4    # Torque constance
air_density = 1.3  # kg/m^3
Cd = 0.32     # Drag coefficient
Cr = 0.01     # Rolling friction coefficient
A = 2.4       # Frontal area, m^2
g = 9.81      # Gravity

# Define fuzzy variables
error = ctrl.Antecedent(np.linspace(-50, 50, 100), 'error')
d_error = ctrl.Antecedent(np.linspace(-10, 10, 100), 'd_error')
throttle = ctrl.Consequent(np.linspace(-0.1, 1.0, 100), 'throttle')

# Membership functions for error
error['negative'] = fuzz.trimf(error.universe, [-50, -25, 0])
error['zero'] = fuzz.trimf(error.universe, [-10, 0, 10])
error['positive'] = fuzz.trimf(error.universe, [0, 25, 50])

# Membership functions for d_error
d_error['negative'] = fuzz.trimf(d_error.universe, [-10, -5, 0])
d_error['zero'] = fuzz.trimf(d_error.universe, [-2, 0, 2])
d_error['positive'] = fuzz.trimf(d_error.universe, [0, 5, 10])

# Membership functions for throttle
throttle['low'] = fuzz.trimf(throttle.universe, [-0.1, 0.0, 0.5])
throttle['medium'] = fuzz.trimf(throttle.universe, [0.2, 0.5, 0.8])
throttle['high'] = fuzz.trimf(throttle.universe, [0.5, 1.0, 1.0])

# Define fuzzy rules
rule1 = ctrl.Rule(error['negative'] & d_error['negative'], throttle['high'])
rule2 = ctrl.Rule(error['negative'] & d_error['zero'], throttle['medium'])
rule3 = ctrl.Rule(error['negative'] & d_error['positive'], throttle['low'])
rule4 = ctrl.Rule(error['zero'] & d_error['negative'], throttle['medium'])
rule5 = ctrl.Rule(error['zero'] & d_error['zero'], throttle['low'])
rule6 = ctrl.Rule(error['positive'] & d_error['positive'], throttle['low'])

# Create control system
throttle_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
throttle_sim = ctrl.ControlSystemSimulation(throttle_ctrl)

# Simulation setup
def simulate_cruise_control(init_speed, target_speed, time_steps):
    speed = init_speed
    speeds = []
    throttles = []
    throttle_position = 0

    for t in range(time_steps):
        # Calculate forces
        Fd = 0.5 * air_density * Cd * A * speed ** 2
        Fr = mass * g * Cr * np.sign(speed)
        T = Tm * (1 - beta * ((alpha_n * speed) / omega_n - 1) ** 2)
        Fe = alpha_n * throttle_position * T

        # Dynamics
        acceleration = (Fe - Fd - Fr) / mass
        speed += acceleration * 0.1  # Update speed

        # Fuzzy control
        speed_error = target_speed - speed
        d_speed_error = -acceleration

        # Clip inputs to universe of discourse
        speed_error = np.clip(speed_error, error.universe[0], error.universe[-1])
        d_speed_error = np.clip(d_speed_error, d_error.universe[0], d_error.universe[-1])

        # Debug inputs
        # print(f"Speed Error: {speed_error}, Rate of Error: {d_speed_error}")

        # Compute fuzzy output
        try:
            throttle_sim.input['error'] = speed_error
            throttle_sim.input['d_error'] = d_speed_error
            throttle_sim.compute()
            throttle_position = throttle_sim.output['throttle']
        except KeyError:
            # Handle undefined output
            throttle_position = 0.0
            # print("Warning: Fuzzy system could not compute a valid output, using default throttle position.")

        # Store results
        speeds.append(speed)
        throttles.append(throttle_position)

    return speeds, throttles

# Run simulation
time_steps = 10000
init_speed = 20/3.6  # m/s
target_speed = 100/3.6  # m/s
speeds, throttles = simulate_cruise_control(init_speed, target_speed, time_steps)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, time_steps * 0.1, time_steps), speeds, label='Vehicle Speed')
plt.axhline(target_speed, color='r', linestyle='--', label='Target Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Cruise Control Simulation with Fuzzy Controller')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, time_steps * 0.1, time_steps), throttles, label='Throttle Position', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Throttle Position')
plt.title('Throttle Position Over Time')
plt.legend()
plt.grid()
plt.show()