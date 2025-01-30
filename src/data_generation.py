from gates import feature_gate
from qiskit.quantum_info import random_unitary, Pauli
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import random
import math
import numpy as np

# PARAMETERS OF THE CIRCUIT
n = 2 # n_qubits
d = 2 # input dimension

def assign_label(x1, x2, delta, V):
    # Creation of circuit
    feat = feature_gate(x1, x2)
    #V = UnitaryGate(random_unitary(2**n))
    qc = QuantumCircuit(n)
    qc.append(feat, range(n))
    qc.append(V, range(n))

    # Expectation value of f=ZZ
    f = Pauli('ZZ')
    estimator = Estimator()
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=AerSimulator())
    qc_transpiled = pass_manager.run(qc)
    job = estimator.run([(qc_transpiled, f)])
    result = job.result()[0]
    exp_value = float(result.data.evs)

    # Return the label (if exp_value doesn't match, return 0)
    if exp_value <= -delta:
        return -1
    if exp_value >= delta:
        return 1
    return 0
        
def generate_data(n_data, delta, seed=35):
    data = []
    while len(data) < n_data:
        x1 = random.random() * 2 * math.pi
        x2 = random.random() * 2 * math.pi
        label = assign_label(x1, x2, delta)
        if label != 0:
            data.append({'data': [x1, x2], 'label': label})
    return data

"""
Generates a labeled grid of points (returns the list of dicts too)
"""
def generate_grid(grid_size, delta, seed=35):
    grid = np.zeros((grid_size, grid_size))
    grid_ds = []
    V = UnitaryGate(random_unitary(2**n))
    for i in range(grid_size):
        x = 2*math.pi*i/grid_size
        for j in range(grid_size):
            y = 2*math.pi*j/grid_size
            grid[i, j] = assign_label(x, y, delta, V)
            #grid_ds.append({'input': [x, y], 'label': grid[i, j]})
    return {'unitary': V.to_matrix(), 'grid_size': grid_size, 'delta': delta, 'grid': grid}

