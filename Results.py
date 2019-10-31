# %%
from Circuit_ops import read_fidelities, plot_fidelities

# %%
# Read the fidelities given the file name
N = 4
num_gates = 159
fn = f'fidelities_{N}_qubits_{num_gates}_gates.txt'
fidelities = read_fidelities(fn)
runs = len(fidelities)
title = f'Fidelity distribution for {N} ions, {num_gates} gate, {runs} runs'
plot_fidelities(fidelities, title)

# %%
N = 9
num_gates = 160
fn = f'fidelities_{N}_qubits_{num_gates}_gates.txt'
fidelities = read_fidelities(fn)
runs = len(fidelities)
title = f'Fidelity distribution for {N} ions, {num_gates} gate, {runs} runs'
plot_fidelities(fidelities, title)

# %%
