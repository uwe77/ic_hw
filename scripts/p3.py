import numpy as np
from pso import PSO


# Define Ackley function
def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)



# Run PSO for Ackley
pso = PSO(ackley, dimensions=2)
best_position = pso.optimize()
print("Best position for Ackley:", best_position)
pso.plot_best_scores()