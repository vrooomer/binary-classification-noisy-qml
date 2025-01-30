import math
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
from qiskit.circuit.library import HamiltonianGate, UnitaryGate, RZGate, RZZGate
import numpy as np

# PARAMETERS OF THE CIRCUIT
n = 2 # n_qubits
d = 2 # input dimension

# ====================================================== FEATURE SPACE
# FEATURE MAPS
def phi_1(x1, x2):
    return x1

def phi_2(x1, x2):
    return x2

def phi_12(x1, x2):
    return (math.pi - x1)*(math.pi - x2)

# DIAG GATE CREATION
def diag_gate(x1, x2):
    qc = QuantumCircuit(n)
    qc.rzz(-2*phi_12(x1, x2), 0, 1)
    qc.rz(-2*phi_2(x1, x2), 1)
    qc.rz(-2*phi_1(x1, x2), 0)
    return qc.to_gate(label=r'$U_{\Phi(\vec x)}$')

# FEATURE GATE CREATION
def feature_gate(x1, x2):
    diag = diag_gate(x1, x2)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.append(diag, range(n))
    qc.h(range(n))
    qc.append(diag, range(n))
    #qc.draw(output='mpl')
    return qc.to_gate(label=r'$\mathcal{U}_{\Phi(\vec x)}$')


# ====================================================== PARAMETRIC CIRCUIT
# SINGLE QUBIT PARAMETRIC ROTATIONS
def local_rot_gate(theta_z, theta_y, label=r'$\theta$'):
    qc = QuantumCircuit(1)
    qc.ry(-theta_y, 0)
    qc.rz(-theta_z, 0)
    return qc.to_gate(label=label)
    
