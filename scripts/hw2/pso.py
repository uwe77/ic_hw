import numpy as np
from matplotlib import pyplot as plt

# Particle Swarm Optimization (PSO)
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# Particle Swarm Optimization with plotting
class PSO:
    def __init__(self, func, num_particles=30, dimensions=2, w=0.5, c1=2, c2=2):
        self.func = func
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.positions = np.random.uniform(-5, 5, (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.array([func(p) for p in self.positions])
        self.global_best_position = self.best_positions[np.argmin(self.best_scores)]
        self.global_best_scores = []
    
    def optimize(self, iterations=100):
        for _ in range(iterations):
            for i, position in enumerate(self.positions):
                score = self.func(position)
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = position
                if score < self.func(self.global_best_position):
                    self.global_best_position = position
            self.global_best_scores.append(self.func(self.global_best_position))
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.best_positions[i] - self.positions[i]) +
                    self.c2 * r2 * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
        return self.global_best_position

    def plot_best_scores(self):
        plt.plot(self.global_best_scores)
        plt.title("PSO Optimization Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Best Score (Ackley Function)")
        plt.show()
