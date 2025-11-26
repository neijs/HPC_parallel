import numpy as np
import matplotlib.pyplot as plt
import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
local_M = int(sys.argv[3])
local_N = int(sys.argv[4])
save = int(sys.argv[5])

# Read CSV files
cg_result = np.genfromtxt('cg_result.csv', delimiter=',', dtype=float)
cg_energy_func = np.genfromtxt('cg_energy_func.csv', delimiter=',', dtype=float)
cg_error  = np.genfromtxt('cg_error.csv',  delimiter=',', dtype=float)

# Create heatmap
size = local_M*local_N
proc_rows = M//local_M
proc_cols = N//local_N
num_procs = proc_rows*proc_cols
domain = []
for i in range(proc_rows):
    row = []
    for j in range(proc_cols):
        idx = i*proc_cols + j
        start = idx*size
        end = start + size
        subdomain = cg_result[start:end].reshape(local_M, local_N)
        row.append(subdomain)
    domain.append(row)
result = np.block(domain)

plt.figure(1)
plt.imshow(np.flipud(result.T))
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
plt.plot(cg_error.flatten())
plt.title('Error curve')
plt.xlabel('Iteration')
if save == 1:
    plt.savefig(f'Error_{M}_{N}.png', dpi=300)
plt.show()