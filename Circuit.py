from Circuit_ops import apply, zero_state
from Compiler import compile_gates

# -- The quantum circuit --

class Circuit:
    def __init__(self, N):
        # Circuit name
        self.name = "Circuit"
        # Number of qubits
        self.N = N
        # Single-qubit over-rotation, phase-error, two-qubit-over-rotation
        self.errors = [.0, .0, .0]
        # Ideal gates used for plotting circuits
        self.ideal_gates = []
        # Native gates (e.g. in ion traps)
        self.native_gates = []
        # Noisy native gates: [bases, qubits, rotation angle, axis angle]
        self.noisy_gates = []
        # Number of circuit execution with randomized gate noises
        self.runs = 2
        # Initialize state to be zero state
        self.init_state = zero_state(N)
        
        self.Z_is_native = False

        self.library = "cupy"

    def compile_circuit_gates(self):
        errors = self.errors
        ideal_gates = self.ideal_gates
        runs = self.runs
        Z_is_native = self.Z_is_native

        native_gates, noisy_gates = compile_gates(ideal_gates, errors, runs, Z_is_native=Z_is_native)

        self.native_gates = native_gates
        self.noisy_gates = noisy_gates

    # Computes the final state given the initial state and circuit
    def compute(self, mode="single-init-state", compile_gates=True):
        # Run circuit with different noise distributions given one initial state
        if mode == "single-init-state":
            # Make sure number of runs is 2 or larger
            runs = self.runs if self.runs > 1 else 2
            # Clone initial state for multiple runs with different gate errors
            try:
                from cupy import tile
            except ImportError:
                from numpy import tile
            states = tile(self.init_state, (runs, 1))
        # Run the same circuit given multiple initial states
        elif mode == "mulitple-init-states":
            self.runs = len(self.init_state)
            states = self.init_state
        
        if compile_gates:
            self.compile_circuit_gates()

        noisy_gates = self.noisy_gates
        
        for gate in noisy_gates:
            states = apply(gate, states)
        return states

    # Clear gates
    def clear_gates(self):
        self.ideal_gates = []
        self.noisy_gates = []

    # -- Ideal gates in a circuit --
    def S_phi(self, q, t, phi):
        self.ideal_gates.append(["S_phi", q, t, phi])
        return self

    def X(self, q, t):
        self.ideal_gates.append(["X", q, t, None])
        return self

    def Y(self, q, t):
        self.ideal_gates.append(["Y", q, t, None])
        return self

    def Z(self, q, t):
        self.ideal_gates.append(["Z", q, t, None])
        return self

    def XX(self, q1, q2, t):
        self.ideal_gates.append(["XX", [q1, q2], t, None])
        return self

    # -- Synthesized gates --
    def H(self, q):
        self.ideal_gates.append(["H", q, None, None])
        return self

    def CNOT(self, q1, q2):
        self.ideal_gates.append(["CNOT", [q1, q2], None, None])
        return self

    # -- Plot circuit for ideal gates --
    def plot_circuit(self):
        return