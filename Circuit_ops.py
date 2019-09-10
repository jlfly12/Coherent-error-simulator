from Gate_bases import x, y, z, s_phi, identity
import matplotlib.pyplot as plt
import time
import numpy as np

try:
    import cupy as cp
    from cupy import rint, array, pi, exp, log2, sin, cos, random, linspace, sort, copy, zeros, roll, swapaxes, multiply, matmul, angle, binary_repr
    from cupy import sum as arr_sum
    from cupy import absolute as arr_abs

except ImportError:
    from numpy import rint, array, pi, exp, log2, sin, cos, random, linspace, sort, copy, zeros, roll, swapaxes, multiply, matmul, angle, binary_repr
    from numpy import sum as arr_sum
    from numpy import absolute as arr_abs

# Check if cupy is imported
def check_if_cupy_is_imported():
    import sys
    return True if 'cupy' in sys.modules else False


# Convert state index to a bit string (useful for extracting specific state amplitudes)
def int_to_bit_str(integer, N):
    return array(list(binary_repr(integer, width=N)), dtype=int)

# Do the opposite
def bit_str_to_int(bit_str):
    return int(''.join(str(e) for e in bit_str), 2)

def zero_state(n):
    state = zeros(2 ** n, dtype=complex)
    state[0] = 1
    return state

def remove_global_phase(states):
    gp = exp(- 1j * angle(states[0]))
    c = swapaxes(states, 0, 1)
    states = multiply(gp.T, c).T
    return states

# -- Apply gate to state --


def apply(gate, states, global_phase=False):

    # A shorthand for the original states
    a = states
    # d1 = number of circuit runs with noise, d2 = 2 ** N = dimension of state vector

    d1, d2 = states.shape
    N = int(rint(log2(d2)))

    # A copy of state a, to be flipped by qubit-wise Pauli operations
    b = copy(a)

    # print("d1 = ", d1)
    # print("d2 = ", d2)
    # print("N = ", N)
    # Reshape to rank-(N+1) tensor
    b = b.reshape([d1] + [2] * N)

    for k in range(len(gate[0])):

        basis = gate[0][k]
        q = gate[1][k]

        if basis == identity:
            pass

        if basis == x:
            b = roll(b, 1, q+1)

        if basis == y:
            b = roll(b, 1, q+1)
            b = swapaxes(b, 0, q+1)
            b[0] *= -1j
            b[1] *= 1j
            b = swapaxes(b, 0, q+1)

        if basis == s_phi:
            phi = array(gate[3][k])
            b = roll(b, 1, q+1)
            b = swapaxes(b, 0, q+1)
            b = swapaxes(b, N, q+1)
            phase1 = cos(phi) + 1j * sin(phi)
            phase2 = cos(phi) - 1j * sin(phi)
            b[0] = multiply(phase2, b[0])
            b[1] = multiply(phase1, b[1])
            b = swapaxes(b, N, q+1)
            b = swapaxes(b, 0, q+1)

        if basis == z:
            b = swapaxes(b, 0, q+1)
            b[1] *= -1
            b = swapaxes(b, 0, q+1)

    b = b.reshape(d1, d2)
    angles = array(gate[2][0])

    states = (cos(angles/2) * a.T - 1j * sin(angles/2) * b.T).T

    # Remove global phase (may be awkward if first amplitude is close to zero)

    if global_phase == False:
        pass

    return states

# Plot state probabilities for one state vector only
# ADD COLOUR FOR PHASE?

def state_prob_plot(state, title='State probability plot'):
    if check_if_cupy_is_imported():
        state = cp.asnumpy(state)
    for i in range(len(state)):
        plt.title(title)
        plt.plot(np.linspace(i, i, 2), np.linspace(
            0, np.absolute(state[i]) ** 2, 2), 'b', linewidth=5)
    plt.show()

# Calculate, plot, save and read fidelities

def find_fidelities(states, ideal):
    if states.shape == ideal.shape:
        
        fidelities = arr_abs(arr_sum(states * ideal.conj(), axis=1)) ** 2
    else:
        fidelities = arr_abs(matmul(states, ideal.conj())) ** 2
    return fidelities


def plot_fidelities(fidelities, bins=20, range=None, title="Fidelity plot"):
    if check_if_cupy_is_imported():
        fidelities = cp.asnumpy(fidelities)
    runs = len(fidelities)
    plt.title(title)
    plt.hist(fidelities, bins, range)
    plt.show()
    print(f'Average fidelity = {arr_sum(fidelities) / len(fidelities)}')
    print(f'10-th percentile fidelity = {sort(fidelities)[int(runs/10)]}')
    print(f'90-th percentile fidelity = {sort(fidelities)[int(9*runs/10)]}')


def save_fidelities(fidelities, N, n_gates, err, runs, fn="Fidelities"):
    with open(fn, 'w') as f:
        f.write(
            f'Results for {N} qubits, {n_gates} gates with max error = {err * 100}% over {runs} runs \n')
        for fidelity in fidelities:
            f.write("%s\n" % fidelity)
        f.close()


def read_fidelities(fn):
    with open(fn) as f:
        fidelities = []
        F = f.read().splitlines()

        # The first line in the txt file is just a description
        for i in range(len(F)-1):
            fidelities.append(float(F[i+1]))
    return fidelities

    
# Find probability of measuring a K-qubit state given an N-qubit state
def find_prob(measured_qubits, sub_state, states):
    
    # Make sure measured qubit numbers are in ascending order
    qubits = measured_qubits
    qubits.sort()
    
    # Make a copy of given states in order not to alter them
    a = states.copy()
    d1, d2 = a.shape          # d1 = number of circuit runs, d2 = 2 ** N
    N = int(rint(log2(d2)))

    # Reshape to rank-(N+1) tensor
    a = a.reshape([d1] + [2] * N)
    
    # K = number of measured qubits, M = number of qubits not measured
    K = len(qubits)
    M = N - K
    
    # Reorder qubit number axes
    for i in range(K):
        a = swapaxes(a, i + 1, qubits[i] + 1)
    
    # Flatten arrays for 2 groups of qubits
    a = a.reshape([d1] + [2 ** K] + [2 ** M])

    # Broadcast multiply coefficients
    a = swapaxes(a, 0, 1)
    a = multiply(a.T, sub_state).T
    
    # Sum over coefficients
    a = a.sum(axis=0)
    a = abs(a) ** 2
    a = a.sum(axis=1)
    
    # Return probability of measuring a substate for all circuit runs
    return a
    
    
def plot_prob(probabilities, bins=20, range=None, title="Probability of measuring a specific state from subsystem"):
    if check_if_cupy_is_imported():
        probabilities = cp.asnumpy(probabilities)
    runs = len(probabilities)
    plt.title(title)
    plt.hist(probabilities, bins, range)
    plt.show()
    print(f'Average probability = {arr_sum(probabilities) / len(probabilities)}')
    print(f'10-th percentile probability = {sort(probabilities)[int(runs/10)]}')
    print(f'90-th percentile probability = {sort(probabilities)[int(9*runs/10)]}')
    

