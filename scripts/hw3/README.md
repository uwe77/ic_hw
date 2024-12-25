# Intelligent Control Homework 3

This project contains solutions for Homework 3, focusing on designing and implementing intelligent controllers for two tasks. Each solution includes fuzzy logic controllers and a comparison with other methods.

---

## 1. Fuzzy Controller for First-Order Plus Time Delay Model

### Task Description
Design a fuzzy controller to control a system with the transfer function:
\[
G(s) = \frac{e^{-s}}{10s + 1}
\]

This task is split into two parts:

- **(a)** Design a fuzzy controller based on tracking errors and their derivatives for inference.
- **(b)** Compare the fuzzy controller's performance with a PID controller optimized using Particle Swarm Optimization (PSO).

### Implementation
1. **Fuzzy Controller**:
    - **Input Variables**:
        - `Error`: Tracking error.
        - `Derivative of Error`: Rate of change of the error.
    - **Output Variable**:
        - `Controller Output`: Adjustments to reduce error.
    - **Rules**:
        - Nine rules derived from PID-like schemes (e.g., if error is positive and increasing, decrease output).
    - Simulated with sinusoidal error input.

2. **PID Controller**:
    - Optimized using PSO to minimize the sum of squared errors (SSE).
    - Comparison includes step response analysis.

### Results
- **Step Response**: Shows how the fuzzy controller adapts to dynamic inputs.
- **Comparison**: Visualized differences in performance metrics (e.g., SSE).

---

## 2. Cruise Control Using Fuzzy Controller

### Task Description
Design a fuzzy controller for vehicle cruise control. The goal is to maintain a constant cruising speed under varying conditions.

### Implementation
1. **System Dynamics**:
    - Control variable: `Throttle Position`, proportional to fuel injection.
    - Forces modeled: Air resistance, rolling friction, and engine output.

2. **Fuzzy Controller**:
    - **Input Variable**:
        - `Speed Error`: Difference between target and current speed.
    - **Output Variable**:
        - `Throttle Adjustment`: Increase or decrease throttle to maintain speed.
    - **Rules**:
        - Example: If speed error is negative, increase throttle.

3. **Simulation**:
    - Sinusoidal speed error input to test dynamic response.

### Results
- **Throttle Adjustment Response**: Demonstrates how the controller adapts to varying errors to maintain speed.

---

## How to Run
1. Ensure all dependencies are installed:
    - Python libraries: `numpy`, `matplotlib`, `scikit-fuzzy`.
2. Run the script:
    ```bash
    python hw3.py
    ```
3. View plots for both controllers' performance.

---

## Dependencies
- **Python**: Version 3.8+
- **Libraries**:
    - `numpy`
    - `matplotlib`
    - `scikit-fuzzy`

---

## File Structure
- `hw3.py`: Main script containing both tasks.
- `pso.py`: Implementation of the PSO algorithm for PID optimization.

---

## Outputs
- Step response for the fuzzy controller.
- Comparison plots between the fuzzy controller and the optimized PID controller.
- Throttle response for the cruise control system.

---

## Author
This solution was developed for Intelligent Control Homework 3, demonstrating the application of fuzzy logic and optimization techniques in control systems.

