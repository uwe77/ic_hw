# Intelligent Control Homework 3 - Solution
import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from skfuzzy import membership as mf  # Import the membership functions

# Fuzzy Controller Design for First-Order Plus Time Delay Model
# System Transfer Function: G(s) = e^(-s) / (10s + 1)
def fuzzy_controller_first_order():
    # Define fuzzy variables
    error = ctrl.Antecedent(np.linspace(-1, 1, 51), 'error')
    d_error = ctrl.Antecedent(np.linspace(-1, 1, 51), 'd_error')
    output = ctrl.Consequent(np.linspace(-1, 1, 51), 'output')

    # Membership functions
    for var in [error, d_error, output]:
        var['negative'] = mf.trapmf(var.universe, [-1, -1, -0.5, 0])
        var['zero'] = mf.trimf(var.universe, [-0.5, 0, 0.5])
        var['positive'] = mf.trapmf(var.universe, [0, 0.5, 1, 1])

    # Fuzzy rules
    rules = [
        ctrl.Rule(error['negative'] & d_error['negative'], output['negative']),
        ctrl.Rule(error['negative'] & d_error['zero'], output['negative']),
        ctrl.Rule(error['negative'] & d_error['positive'], output['zero']),
        ctrl.Rule(error['zero'] & d_error['negative'], output['negative']),
        ctrl.Rule(error['zero'] & d_error['zero'], output['zero']),
        ctrl.Rule(error['zero'] & d_error['positive'], output['positive']),
        ctrl.Rule(error['positive'] & d_error['negative'], output['zero']),
        ctrl.Rule(error['positive'] & d_error['zero'], output['positive']),
        ctrl.Rule(error['positive'] & d_error['positive'], output['positive'])
    ]

    # Controller system
    fuzzy_system = ctrl.ControlSystem(rules)
    fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_system)

    # Simulate
    time = np.linspace(0, 20, 201)
    step_response = []
    for t in time:
        fuzzy_sim.input['error'] = np.sin(0.1 * t)  # Example error input
        fuzzy_sim.input['d_error'] = 0.1 * np.cos(0.1 * t)  # Example derivative of error
        fuzzy_sim.compute()
        step_response.append(fuzzy_sim.output['output'])

    # Plot results
    plt.plot(time, step_response)
    plt.title("Fuzzy Controller Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Controller Output")
    plt.show()

# Cruise Control with Fuzzy Controller
def fuzzy_controller_cruise():
    # Define fuzzy variables
    speed_error = ctrl.Antecedent(np.linspace(-10, 10, 51), 'speed_error')
    throttle_change = ctrl.Consequent(np.linspace(-0.1, 0.1, 51), 'throttle_change')

    # Membership functions
    speed_error['negative'] = mf.trapmf(speed_error.universe, [-10, -10, -5, 0])
    speed_error['zero'] = mf.trimf(speed_error.universe, [-5, 0, 5])
    speed_error['positive'] = mf.trapmf(speed_error.universe, [0, 5, 10, 10])

    throttle_change['decrease'] = mf.trimf(throttle_change.universe, [-0.1, -0.05, 0])
    throttle_change['maintain'] = mf.trimf(throttle_change.universe, [-0.05, 0, 0.05])
    throttle_change['increase'] = mf.trimf(throttle_change.universe, [0, 0.05, 0.1])

    # Fuzzy rules
    rules = [
        ctrl.Rule(speed_error['negative'], throttle_change['increase']),
        ctrl.Rule(speed_error['zero'], throttle_change['maintain']),
        ctrl.Rule(speed_error['positive'], throttle_change['decrease'])
    ]

    # Controller system
    cruise_system = ctrl.ControlSystem(rules)
    cruise_sim = ctrl.ControlSystemSimulation(cruise_system)

    # Simulate
    time = np.linspace(0, 50, 501)
    cruise_response = []
    for t in time:
        cruise_sim.input['speed_error'] = 5 * np.sin(0.1 * t)  # Example speed error
        cruise_sim.compute()
        cruise_response.append(cruise_sim.output['throttle_change'])

    # Plot results
    plt.plot(time, cruise_response)
    plt.title("Fuzzy Cruise Control Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Throttle Change")
    plt.show()

# Execute both controllers
if __name__ == "__main__":
    print("Running fuzzy controller for first-order plus time delay model...")
    fuzzy_controller_first_order()

    print("Running fuzzy cruise control system...")
    fuzzy_controller_cruise()
