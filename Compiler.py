try: 
    from cupy import sqrt, pi, sin, cos, array, zeros, swapaxes

except ImportError:
    from numpy import sqrt, pi, sin, cos, array, zeros, swapaxes


from Gate_bases import *
from Error_dist import error_dist

# -- The gate compiler --
def compile_gates(gates, errors, runs, Z_is_native):

    # -- Helper functions for returning gates --
    # The native bit-flip rotation
    def S_phi(q, t, phi):
        return [[s_phi], [q], t, [phi]]

    def X(q, t):
        return S_phi(q, t, 0)

    def Y(q, t):
        return S_phi(q, t, pi/2)

    # Z-rotations are assumed to be synthesized from X and Y for now
    def Z(q, t):
        if Z_is_native:
            return [[[z], [q], t, [0]]]
        else:
            return [Y(q, pi/2), X(q, t), Y(q, -pi/2)]

    # XX-rotations are native to ion traps through motional sideband coupling
    def XX(q1, q2, t):
        return [[s_phi, s_phi], [q1, q2], t, [0, 0]]

    native_gates = []
    for gate in gates:

        gate_type = gate[0]     # Gate type
        q = gate[1]             # Qubit(s)
        t = gate[2]             # Gate angle
        phi = gate[3]           # Gate axis (on the x-y plane)

        # -- Switch between different gate types --

        if gate_type == s_phi:
            native_gates.append(S_phi(q, t, phi))

        if gate_type == x:
            native_gates.append(X(q, t))

        if gate_type == y:
            native_gates.append(Y(q, t))

        if gate_type == z:
            for gate in Z(q, t):
                native_gates.append(gate)

        if gate_type == xx:
            native_gates.append(XX(q[0], q[1], t))

        # Hadmard gate synthesis
        if gate_type == h:
            native_gates.extend([Y(q, pi/2), X(q, -pi)])

        # CNOT gate synthesis
        if gate_type == cnot:
            native_gates.extend([Y(q[0], pi/2), XX(q[0], q[1], pi/2),
                                 X(q[0], -pi/2), X(q[1], -pi/2), Y(q[0], -pi/2)])

# -- Compile and output list of native gates --

# Native gates to noisy gates
    [single_err, phase_err, xx_err] = errors

    # Noisy gates: bases, qubit numbers, rotation angle, axis angle
    noisy_gates = []
    for native_gate in native_gates:
        basis = native_gate[0]
        qubits = native_gate[1]

        # Two-qubit gate
        if len(basis) == 2:
            angle = [native_gate[2] * (xx_err * array(error_dist(runs)) + 1)]
            axis1 = native_gate[3][0] + phase_err * \
                array(error_dist(runs)) * pi * sqrt(2) / 4
            axis2 = native_gate[3][1] + phase_err * \
                array(error_dist(runs)) * pi * sqrt(2) / 4

            noisy_gates.append([basis, qubits, angle, [axis1, axis2]])
            # noisy_gates[0].append(native_gates[i][0])
            # noisy_gates[1].append(native_gates[i][1])
            # noisy_gates[2].append(native_gates[i][2] *
            #                       (xx_err * error_dist(runs) + 1))
            # noisy_gates[3].append([native_gates[i][3][0] + phase_err * error_dist(runs) * pi * sqrt(2) / 4,
            #                        native_gates[i][3][1] + phase_err * error_dist(runs) * pi * sqrt(2) / 4])
        # Single-qubit gate
        else:
            angle = [native_gate[2] *
                     (single_err * array(error_dist(runs)) + 1)]
            axis = [native_gate[3][0] +
                    phase_err * array(error_dist(runs)) * pi / 2]

            noisy_gates.append([basis, qubits, angle, axis])
            # noisy_gates[0].append(native_gates[i][0])
            # noisy_gates[1].append(native_gates[i][1])
            # noisy_gates[2].append(native_gates[i][2] *
            #                       (single_err * error_dist(runs) + 1))
            # noisy_gates[3].append(native_gates[i][3] +
            #                       phase_err * error_dist(runs) * pi / 2)

    return native_gates, noisy_gates
