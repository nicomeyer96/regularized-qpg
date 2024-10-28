# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _equal_superposition(vqc, num_qubits):
    """ Creates initial equal superposition over all computational basis states
    """
    for q in range(num_qubits):
        vqc.h(q)


def _variational_layer(vqc, parameters_variational, num_qubits, depth):
    """ Variational layer
    """
    for q in range(num_qubits):
        idx = (2 * num_qubits) * depth + 2 * q
        vqc.rz(parameters_variational[idx], q)
        vqc.ry(parameters_variational[idx + 1], q)


def _encoding_layer_with_scaling(vqc, parameters_encoding, parameters_scaling, num_qubits, depth):
    """ Feature map with multiplicative scaling parameters
    """
    for q in range(num_qubits):
        idx = (2 * num_qubits) * depth + 2 * q
        vqc.ry(parameters_scaling[idx] * parameters_encoding[q], q)
        vqc.rz(parameters_scaling[idx+1] * parameters_encoding[q], q)


def _encoding_layer(vqc, parameters_encoding, num_qubits):
    """ Feature map
    """
    for q in range(num_qubits):
        vqc.ry(parameters_encoding[q], q)
        vqc.rz(parameters_encoding[q], q)


def _entanglement_layer_nn_cx(vqc, num_qubits):
    """ CX Entangling layer (nearest neighbor)
    """
    for q in range(num_qubits-1):
        vqc.cx(q, q+1)
    if num_qubits > 2:
        vqc.cx(num_qubits-1, 0)


def _entanglement_layer_full_cz(vqc, num_qubits):
    """ CZ Entangling layer (all-to-all)
    """
    for q in range(num_qubits):
        for qq in range(q+1, num_qubits):
            vqc.cz(q, qq)


def generate_circuit(
        num_qubits: int = 4,
        depth: int = 1,
        entanglement_structure: str = 'nn',
        input_scaling: bool = False
) -> tuple[QuantumCircuit, ParameterVector, tuple[ParameterVector, None | ParameterVector]]:
    """ Generate circuit, optionally with trainable input scaling parameters
    """

    if entanglement_structure not in ['nn', 'full']:
        ValueError('Entanglement structure {} unknown. Choose either `nn` (nearest-neighbors) or `full` (all-to-all)')

    vqc = QuantumCircuit(num_qubits)

    # data encoding (`depth` instances, i.e. data re-uploading for `depth>=2`)
    num_parameters_encoding = num_qubits
    parameters_encoding = ParameterVector('s', length=num_parameters_encoding)

    # optional input scaling
    parameters_scaling = None
    if input_scaling:
        num_parameters_scaling = 2 * num_qubits * depth
        parameters_scaling = ParameterVector('\u03BB', length=num_parameters_scaling)

    # variational parameters (`depth+1` instances)
    num_parameters_variational = 2 * num_qubits * (depth + 1)
    parameters_variational = ParameterVector('\u03B8', length=num_parameters_variational)

    # compose circuit
    _equal_superposition(vqc, num_qubits)
    vqc.barrier()
    for d in range(depth):
        _variational_layer(vqc, parameters_variational, num_qubits, depth=d)
        _entanglement_layer_nn_cx(vqc, num_qubits) if 'nn' == entanglement_structure else _entanglement_layer_full_cz(vqc, num_qubits)
        vqc.barrier()
        _encoding_layer_with_scaling(vqc, parameters_encoding, parameters_scaling, num_qubits, depth=d) \
            if parameters_scaling \
            else _encoding_layer(vqc, parameters_encoding, num_qubits)
        vqc.barrier()
    _variational_layer(vqc, parameters_variational, num_qubits, depth=depth)

    return vqc, parameters_encoding, (parameters_variational, parameters_scaling)
