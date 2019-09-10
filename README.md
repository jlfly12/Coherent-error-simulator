# QRSim (Qubit-wise Rotation Simulator)

Standalone, GPU-capable quantum circuit simulator in less than 500 lines.

Good for:

- Single and 2-qubit gate rotations

All you need:

- numpy (core)
- matplotlib (graph plotting)

Optional:

- cupy (replaces numpy when CUDA-enabled GPU is detected)

Functions:

- Compute final state given a sequence of gates
- Custom gate compiler
- Edit and keep track of gates (ideal/noisy) every step of the way
- Combination of time-dependent and independent noises
- Arbitrary error distribution
- Parallel execution of thousands of circuit runs as far as memory permits
- Fidelity plots
- Probability of measuring any output state

Data flow:

- Gate compilation: ideal gates ---(noise distribution)--> compiled gates (optional) --> noisy gates
- Initial states --> final states

Analysis:

- Fidelity histogram
- Probability of getting desired state
- Fidelity distribution over multiple input states/gate sets

Benchmark:

- Bell state for 10 qubits, 5000 runs


Potential applications

- Evaluate Circuit/Algorithm performance
- Characterize gate errors

Demos:

- IQFT circuit
- 6-qubit parity check

License:
Apache License v2.0
