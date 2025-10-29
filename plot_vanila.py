import numpy as np
import matplotlib.pyplot as plt
import sys

M = sys.argv[1]
N = sys.argv[2]
save = int(sys.argv[3])

# Read CSV files
cg_result = np.genfromtxt('cg_result.csv', delimiter=',', dtype=float)
cg_energy_func = np.genfromtxt('cg_energy_func.csv', delimiter=',', dtype=float)
cg_difference  = np.genfromtxt('cg_difference.csv',  delimiter=',', dtype=float)

# Create heatmap
plt.figure(1)
plt.imshow(np.flipud(cg_result.T))
plt.title('Heatmap of the solution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
if save == 1:
    plt.savefig(f'Solution_{M}_{N}.png', dpi=300)

# Create energy functional curve
plt.figure(2)
plt.plot(cg_energy_func.flatten())
plt.title('Energy functional curve')
plt.xlabel('Iteration')
if save == 1:
    plt.savefig(f'Energy_{M}_{N}.png', dpi=300)

# Create difference curve
plt.figure(3)
plt.plot(cg_difference.flatten())
plt.title('Difference curve')
plt.xlabel('Iteration')
if save == 1:
    plt.savefig(f'Difference_{M}_{N}.png', dpi=300)
plt.show()