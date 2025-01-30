from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score
import sys
from scipy.optimize import minimize, curve_fit

# The number of classes, here +/-
n_classes = 2

# Number of lambdas, lmbd_i=1+2*(i-1)/l
n_lambdas = 3

# Our default parity funtion is f=ZZ, '0' for '+1' eigenvalue, '1' for eigenvalue '-1'
def parity(x):
    return "{:b}".format(x).count("1") % 2

class AerQNN:
    """
    Creates a neural network with a specified sampling process (number of shots, depolarizing noise rate)
    """
    def __init__(self, qc, n_dims, n_shots, noise_rate=0.0, interpret=parity, optimization_level=3):
        sampler = Sampler(default_shots=n_shots)
        if noise_rate > 0.0:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_rate, 2), ['cz']) # So far, only applies to CZ
            sampler = Sampler(default_shots=n_shots, options=dict(backend_options=dict(noise_model=noise_model)))
        
        pm = generate_preset_pass_manager(optimization_level, AerSimulator())
        aer_qc = pm.run(qc)
        self.qnns = []
        self.qnns.append(SamplerQNN(
            circuit=aer_qc,
            input_params=qc.parameters[:n_dims], 
            weight_params=qc.parameters[n_dims:], 
            interpret=interpret, 
            output_shape=n_classes,
            sampler=sampler
        ))
        self.qc = qc
        self.aer_qc = aer_qc
        self.n_shots = n_shots
        self.n_inputs = n_dims
        self.n_weights = len(qc.parameters[n_dims:])
        self.noise_rate = noise_rate
        self.n_layers = self.n_weights/4 - 1

        # More noisy qnns, and lambas, ONLY IF L>0
        self.lambdas = [1]
        if self.n_layers > 0:
            new_qc = qc.copy()
            for i in range(1, n_lambdas):
                self.lambdas.append(1+2*i/self.n_layers)
                new_qc.cz(0, 1)
                new_qc.cz(0, 1)
                pm = generate_preset_pass_manager(0, AerSimulator())
                aer_qc = pm.run(new_qc.copy())
                self.qnns.append(SamplerQNN(
                    circuit=aer_qc,
                    input_params=qc.parameters[:n_dims], 
                    weight_params=qc.parameters[n_dims:], 
                    interpret=interpret, 
                    output_shape=n_classes,
                    sampler=sampler
                ))

    """
    Forward step in QNN, returns the probabilities of +/- classes given a single datapoint
    """
    def forward(self, input_data, weights):
        return self.qnns[0].forward(input_data, weights)
    
    """
    Forward step in QNN, returns the probabilities of +/- classes given a single datapoint, for each artificially noisy circuit
    """
    def lmbd_forward(self, input_data, weights):
        l = []
        for qnn in self.qnns:
            l.append(qnn.forward(input_data, weights))
        return l
    
    """
    Returns the exact expectation of M_+ (the sum of probabilities over all outcomes belonging to the + class)
    """
    def exact_expectation(self, input_data, weights):
        # TODO: parity observable is hardcoded
        m_plus = SparsePauliOp(["II", "ZZ"], coeffs=[0.5, 0.5])
        pm = generate_preset_pass_manager(optimization_level=1)
        isa_circuit = pm.run(self.qc)
        isa_observable = m_plus.apply_layout(isa_circuit.layout)
        estimator = StatevectorEstimator()
        job = estimator.run([(isa_circuit, isa_observable, input_data+weights)])
        result = job.result()
        return result[0].data.evs.tolist()
    
    """
    Using the forward step, predicts the label of the given datapoint
    """
    def predict_label(self, input_data, weights):
        probs = self.forward(input_data, weights)
        return 1 - 2*np.argmax(probs)
    
    """
    Computes the accuracy by computing the confusion matrix between the given labeled data and the predicted data
    """
    def accuracy(self, eval_data, weights):
        # Predict labels
        y_pred = [self.predict_label(d['input'], weights) for d in eval_data]
        # Accuracy
        y_true = [d['label'] for d in eval_data]
        return accuracy_score(y_true, y_pred)
    

# ======================================================================
# ======================================================= TRAINING LOOPS
# ======================================================================
"""
Training loop using the specified method and parameters. Uses the COBYLA optimizer.

Returns a dict.
"""
def training(train_qnn, test_qnn, init_weights, init_bias, training_dataset, testing_dataset, n_iter=200, fun='emp_risk', zne=False):
    logs = {'cost': [], 'acc': [], 'est': [], 'exa': []}
    # Run the training loop
    if fun == 'emp_risk':
        final_x = minimize(
            lambda w: risk_training(train_qnn, test_qnn, w, training_dataset, testing_dataset, logs, zne),
            x0 = [init_bias] + init_weights,
            method = 'COBYLA',
            options = {'maxiter': n_iter, 'tol': 1e-8} # "forces" COBYLA to never stop before reaching n_iter
        )
    else:
        return
    # Final log
    return {
        'initial_weights': init_weights,
        'initial_bias': init_bias,
        'final_weights': final_x.x[1:].tolist(),
        'final_bias': final_x.x[0],
        'cost_log': logs['cost'],
        'acc_log': logs['acc']
        }

def progress_bar(it, cost):
    sys.stdout.flush()
    print(f'ITERATION {it}, COST: {cost}', end='\r')


# === EMPIRICAL RISK
"""
Training loop using the empirical risk. Each iteration runs on all training dataset, 
the accuracy is ran on the testing dataset, each 10 iterations.
"""
def risk_training(train_qnn, test_qnn, all_weights, training_dataset, testing_dataset, logs, zne):
    it = len(logs['cost'])
    bias = all_weights[0]
    weights = all_weights[1:]
    # Computes the risk
    cost = emp_risk(train_qnn, weights, bias, training_dataset, zne)
    logs['cost'].append(cost)
    # Computes accuracy (each 10 steps)
    if it%10 == 0:
        acc = test_qnn.accuracy(testing_dataset, weights)
        logs['acc'].append(acc)
    progress_bar(it+1, cost)
    return cost


"""
Computes the empirical risk on the given dataset and parameters.
"""
def emp_risk(train_qnn, weights, bias, training_dataset, zne=False):
    cost = 0.0
    # Forward step for all training dataset
    for d in training_dataset:
        # Forward QNN pass
        if zne:
            p_plus_list = [p[0][0] for p in train_qnn.lmbd_forward(d['input'], weights)]
            try:
                params, _ = curve_fit(
                    lambda l, a, b: a*np.exp(-b*l)+0.5, 
                    train_qnn.lambdas,
                    p_plus_list
                    )
                a, _ = params
                p_plus = max(a + 0.5, 0)
                p_plus = min(p_plus, 1)
            except:
                p_plus = p_plus_list[0]

        else:
            p_plus = train_qnn.forward(d['input'], weights)[0][0]
        y = d['label']
        py = p_plus if y > 0 else 1 - p_plus
        # Compute the cost
        num = np.sqrt(train_qnn.n_shots)*(0.5 - py + 0.5*y*bias)
        den = np.sqrt(2*(1-py)*py)
        if den == 0:
            cost += (np.sign(num)+1)*0.5 # 0 if negative, 1 else
        else:
            cost += expit(num/den)
    cost = cost / len(training_dataset)
    return cost



        
