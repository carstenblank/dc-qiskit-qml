# -*- coding: utf-8 -*-

# Copyright 2018 Carsten Blank
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""

QmlBinaryDataStateCircuitBuilder
=======================================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state._QmlBinaryDataStateCircuitBuilder


.. autosummary::
    :nosignatures:

    QmlBinaryDataStateCircuitBuilder

QmlBinaryDataStateCircuitBuilder
#########################################

.. autoclass:: QmlBinaryDataStateCircuitBuilder
    :members:

"""
import logging
from math import sqrt
from typing import List

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import h
from scipy import sparse

from . import QmlStateCircuitBuilder
from .cnot import CCXFactory

log = logging.getLogger('QubitEncodingClassifierStateCircuit')


class QmlBinaryDataStateCircuitBuilder(QmlStateCircuitBuilder):
    """
    From binary training and testing data creates the quantum state vector and applies a quantum algorithm to create a circuit.
    """

    def __init__(self, ccx_factory, do_optimizations = True):
        # type: (QmlBinaryDataStateCircuitBuilder, CCXFactory, bool) -> None
        """
        Creates the uniform amplitude state circuit builder

        :param ccx_factory: The multiple-controlled X-gate factory to be used
        """
        self.do_optimizations = do_optimizations
        self.ccx_factory = ccx_factory  # type:CCXFactory

    def build_circuit(self, circuit_name, X_train, y_train, X_input):
        # type: (QmlBinaryDataStateCircuitBuilder, str, List[sparse.dok_matrix], any, sparse.dok_matrix) -> QuantumCircuit
        """
        Build a circuit that encodes the training (samples/labels) and input data sets into a quantum circuit.
        Sample data must be given as a binary (sparse) vector, i.e. each vector's entry must be either 0.0 or 1.0.
        It may also be given already normed to unit length instead of binary.

        :param circuit_name: The name of the quantum circuit
        :param X_train: The training data set
        :param y_train: the training class label data set
        :param X_input: the unclassified input data vector
        :return: The circuit containing the gates to encode the input data
        """
        log.debug("Preparing state.")
        log.debug("Raw Input Vector: %s" % X_input)

        # map the training samples and test input to unit length

        def normalizer(x):
            # type: (sparse.dok_matrix) -> sparse.dok_matrix
            norm = sqrt(sum([abs(e)**2 for e in x.values()]))
            for k in x.keys():
                x[k] = x[k]/norm
            return x

        X_train = [normalizer(x) for x in X_train]
        X_input = normalizer(X_input)

        # Calculate dimensions and qubit usage
        count_of_samples, sample_space_dimension = len(X_train), max([s.get_shape()[0] for s in X_train + [X_input]])
        count_of_distinct_classes = len(set(y_train))

        index_of_samples_qubits_needed = (count_of_samples - 1).bit_length()
        sample_space_dimensions_qubits_needed = (sample_space_dimension - 1).bit_length()
        ancilla_qubits_needed = 1
        label_qubits_needed = (count_of_distinct_classes - 1).bit_length() if count_of_distinct_classes > 1 else 1

        log.info("Qubit map: index=%d, ancillary=%d, feature=%d, label=%d", index_of_samples_qubits_needed,
                 ancilla_qubits_needed, sample_space_dimensions_qubits_needed, label_qubits_needed)

        # Create Registers
        ancilla = QuantumRegister(ancilla_qubits_needed, "a")
        index = QuantumRegister(index_of_samples_qubits_needed, "i")
        data = QuantumRegister(sample_space_dimensions_qubits_needed, "f^S")
        qlabel = QuantumRegister(label_qubits_needed, "l^q")

        clabel = ClassicalRegister(label_qubits_needed, "l^c")
        branch = ClassicalRegister(1, "b")

        # Create the Circuit
        qc = QuantumCircuit(ancilla, index, data, qlabel, clabel, branch, name=circuit_name)

        # ======================
        # Build the circuit now
        # ======================

        # Superposition on ancilla & index
        h(qc, ancilla)
        h(qc, index)

        # Create multi-CNOTs
        # First on the sample, then the input and finally the label
        ancilla_and_index_regs = [ancilla[i] for i in range(ancilla.size)] + [index[i] for i in range(index.size)]
        for index_sample, (sample, label) in enumerate(zip(X_train, y_train)):
            cnot_type_sample = (index_sample << 1) + 0
            cnot_type_input = (index_sample << 1) + 1

            # The sample will be encoded
            for basis_vector_index, _ in sample.keys():
                bit_string = "{:b}".format(basis_vector_index)
                for i, v in enumerate(reversed(bit_string)):
                    if v == "1":
                        self.ccx_factory.ccx(qc,
                            cnot_type_sample,
                            ancilla_and_index_regs,
                            data[i])
            # Label will be encoded
            bit_string = "{:b}".format(label)
            for i, v in enumerate(reversed(bit_string)):
                if v == "1":
                    self.ccx_factory.ccx(qc,
                                         cnot_type_sample,
                                         ancilla_and_index_regs,
                                         qlabel[i])

            # The input will be encoded
            for basis_vector_index, _ in X_input.keys():
                bit_string = "{:b}".format(basis_vector_index)
                for i, v in enumerate(reversed(bit_string)):
                    if v == "1":
                        self.ccx_factory.ccx(qc, cnot_type_input, ancilla_and_index_regs, data[i])
            # Label will be encoded
            bit_string = "{:b}".format(label)
            for i, v in enumerate(reversed(bit_string)):
                if v == "1":
                    self.ccx_factory.ccx(qc, cnot_type_input, ancilla_and_index_regs, qlabel[i])

        stop = False
        while not stop and self.do_optimizations:
            dag = circuit_to_dag(qc)
            dag.remove_all_ops_named("barrier")
            ccx_gates = dag.get_named_nodes("ccx")
            cx_gates = dag.get_named_nodes("cx")
            x_gates = dag.get_named_nodes("x")
            removable_nodes = []
            for ccx_gate in ccx_gates.union(x_gates).union(cx_gates):
                successor = list(dag.multi_graph.successors(ccx_gate))
                if len(successor) == 1:
                    node = dag.multi_graph.node[ccx_gate]
                    successor_node = dag.multi_graph.node[successor[0]]
                    if successor_node["name"] == node["name"]:
                        removable_nodes.append(successor[0])
                        removable_nodes.append(ccx_gate)
                    print(end='')
            for n in removable_nodes:
                dag._remove_op_node(n)
            if len(removable_nodes) > 0:
                qc = dag_to_circuit(dag)
            else:
                stop = True

        return qc

    def is_classifier_branch(self, branch_value):
        # type: (QmlBinaryDataStateCircuitBuilder, int) -> bool
        """
        The branch of quantum state bearing the classification is defined to be 0.
        This functions checks this.

        :param branch_value: The measurement of the branch
        :return: True is the measured branch is 0, False if not
        """
        return branch_value == 0
