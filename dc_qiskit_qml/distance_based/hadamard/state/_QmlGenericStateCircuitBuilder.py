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
QmlGenericStateCircuitBuilder
==============================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state._QmlGenericStateCircuitBuilder

The generic state circuit builder classically computes the necessary quantum state vector (sparse) and will use a
state preparing quantum routine that takes the state vector and creates a circuit. This state preparing quantum routine
must implement :py:class:`QmlSparseVectorFactory`.

.. autosummary::
    :nosignatures:

    QmlGenericStateCircuitBuilder

QmlGenericStateCircuitBuilder
##############################

.. autoclass:: QmlGenericStateCircuitBuilder
    :members:

"""
import logging
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from scipy import sparse

from ._QmlStateCircuitBuilder import QmlStateCircuitBuilder
from .sparsevector import QmlSparseVectorStatePreparation

log = logging.getLogger('QmlGenericStateCircuitBuilder')


class RegisterSizes:
    count_of_samples: int
    index_of_samples_qubits: int
    sample_space_dimensions_qubits: int
    ancilla_qubits: int
    label_qubits: int
    total_qubits: int

    def __init__(self, count_of_samples, index_of_samples_qubits, sample_space_dimensions_qubits,
                 ancilla_qubits, label_qubits, total_qubits):
        # type: (int, int, int, int, int, int) -> None
        self.count_of_samples = count_of_samples
        self.index_of_samples_qubits = index_of_samples_qubits
        self.sample_space_dimensions_qubits = sample_space_dimensions_qubits
        self.ancilla_qubits = ancilla_qubits
        self.label_qubits = label_qubits
        self.total_qubits = total_qubits


class QmlGenericStateCircuitBuilder(QmlStateCircuitBuilder):
    """
    From generic training and testing data creates the quantum state vector and applies a quantum algorithm to create
    a circuit.
    """

    def __init__(self, state_preparation):
        # type: (QmlGenericStateCircuitBuilder, QmlSparseVectorStatePreparation) -> None
        """
        Create a new object, use the state preparation routine

        :param state_preparation: The quantum state preparation routine to encode the quantum state
        """
        self.state_preparation = state_preparation
        self._last_state_vector = None  # type: sparse.dok_matrix

    @staticmethod
    def get_binary_representation(sample_index, sample_label, entry_index, is_input, register_sizes):
        # type: (int, int, int, bool, RegisterSizes) -> str
        """
        Computes the binary representation of the quantum state as `str` given indices and qubit lengths

        :param sample_index: the training data sample index
        :param sample_label: the training data label
        :param entry_index: the data sample vector index
        :param is_input: True if the we encode the input instead of the training vector
        :param register_sizes: qubits needed for the all registers
        :return: binary representation of which the quantum state being addressed
        """
        sample_index_b = "{0:b}".format(sample_index).zfill(register_sizes.index_of_samples_qubits)
        sample_label_b = "{0:b}".format(sample_label).zfill(register_sizes.label_qubits)
        ancillary_b = '0' if is_input else '1'
        entry_index_b = "{0:b}".format(entry_index).zfill(register_sizes.sample_space_dimensions_qubits)
        # Here we compose the qubit, the ordering will be essential
        # However keep in mind, that the order is LSB
        qubit_composition = [
            sample_label_b,
            entry_index_b,
            sample_index_b,
            ancillary_b
        ]
        return "".join(qubit_composition)

    @staticmethod
    def _get_register_sizes(X_train, y_train):
        # type: (List[sparse.dok_matrix], np.ndarray) -> RegisterSizes
        count_of_samples, sample_space_dimension = len(X_train), X_train[0].get_shape()[0]
        count_of_distinct_classes = len(set(y_train))

        index_of_samples_qubits_needed = (count_of_samples - 1).bit_length()
        sample_space_dimensions_qubits_needed = (sample_space_dimension - 1).bit_length()
        ancilla_qubits_needed = 1
        label_qubits_needed = (count_of_distinct_classes - 1).bit_length() if count_of_distinct_classes > 1 else 1
        total_qubits_needed = (index_of_samples_qubits_needed + ancilla_qubits_needed
                               + sample_space_dimensions_qubits_needed + label_qubits_needed)
        return RegisterSizes(
            count_of_samples,
            index_of_samples_qubits_needed,
            sample_space_dimensions_qubits_needed,
            ancilla_qubits_needed,
            label_qubits_needed,
            total_qubits_needed
        )

    @staticmethod
    def assemble_state_vector(X_train, y_train, X_input):
        # type: (List[sparse.dok_matrix], any, sparse.dok_matrix) -> sparse.dok_matrix
        """
        This method assembles the state vector for given data. The X data for training (which is a list of sparse
        vectors) is encoded into a sub-space, in the other orthogonal sub-space the same input is encoded everywhere.
        Also, the label is encoded.

        A note: this method is used in conjunction with a generic state preparation, and this may not be the most
        efficient way.

        :param X_train: The training data set
        :param y_train: the training class label data set
        :param X_input: the unclassified input data vector
        :return: The
        """
        register_sizes = QmlGenericStateCircuitBuilder._get_register_sizes(X_train, y_train)

        state_vector = sparse.dok_matrix((2 ** register_sizes.total_qubits, 1), dtype=complex)  # type: sparse.dok_matrix

        factor = 2 * register_sizes.count_of_samples * 1.0

        for index_sample, (sample, label) in enumerate(zip(X_train, y_train)):
            for (i, j), sample_i in sample.items():
                qubit_state = QmlGenericStateCircuitBuilder.get_binary_representation(
                    index_sample, label, i, is_input=False, register_sizes=register_sizes
                )
                state_index = int(qubit_state, 2)
                log.debug("Sample Entry: %s (%d): %.2f.", qubit_state, state_index, sample_i)
                state_vector[state_index, 0] = sample_i

            for (i, j), input_i in X_input.items():
                qubit_state = QmlGenericStateCircuitBuilder.get_binary_representation(
                    index_sample, label, i, is_input=True, register_sizes=register_sizes
                )
                state_index = int(qubit_state, 2)
                log.debug("Input Entry: %s (%d): %.2f.", qubit_state, state_index, input_i)
                state_vector[state_index, 0] = input_i

        state_vector = state_vector * (1 / np.sqrt(factor))

        return state_vector

    def build_circuit(self, circuit_name, X_train, y_train, X_input):
        # type: (QmlGenericStateCircuitBuilder, str, List[sparse.dok_matrix], any, sparse.dok_matrix) -> QuantumCircuit
        """
        Build a circuit that encodes the training (samples/labels) and input data sets into a quantum circuit.

        It does so by iterating through the training data set with labels and constructs upon sample index and
        vector position the to be modified amplitude. The state vector is stored into a sparse matrix of
        shape (n,1) which is stored and can be accessed through :py:func:`get_last_state_vector` for
        debugging purposes.

        Then the routine uses a :py:class:`QmlSparseVectorStatePreparation` routine to encode the calculated state
        vector into a quantum circuit.

        :param circuit_name: The name of the quantum circuit
        :param X_train: The training data set
        :param y_train: the training class label data set
        :param X_input: the unclassified input data vector
        :return: The circuit containing the gates to encode the input data
        """
        log.debug("Preparing state.")
        log.debug("Raw Input Vector: %s" % X_input)

        register_sizes = QmlGenericStateCircuitBuilder._get_register_sizes(X_train, y_train)
        log.info("Qubit map: index=%d, ancillary=%d, feature=%d, label=%d", register_sizes.index_of_samples_qubits,
                 register_sizes.ancilla_qubits, register_sizes.sample_space_dimensions_qubits,
                 register_sizes.label_qubits)

        ancilla = QuantumRegister(register_sizes.ancilla_qubits, "a")
        index = QuantumRegister(register_sizes.index_of_samples_qubits, "i")
        data = QuantumRegister(register_sizes.sample_space_dimensions_qubits, "f^S")
        qlabel = QuantumRegister(register_sizes.label_qubits, "l^q")
        clabel = ClassicalRegister(register_sizes.label_qubits, "l^c")
        branch = ClassicalRegister(1, "b")

        qc = QuantumCircuit(ancilla, index, data, qlabel, clabel, branch, name=circuit_name)  # type: QuantumCircuit

        self._last_state_vector = QmlGenericStateCircuitBuilder.assemble_state_vector(X_train, y_train, X_input)
        return self.state_preparation.prepare_state(qc, self._last_state_vector)

    def is_classifier_branch(self, branch_value):
        # type: (QmlGenericStateCircuitBuilder, int) -> bool
        """
        As each state preparation algorithm uses a unique layout. Here the :py:class:`QmlSparseVectorFactory`
        is asked how the branch for post selection can be identified.

        :param branch_value: The measurement of the branch
        :return: True is the branch is containing the classification, False if not
        """
        return self.state_preparation.is_classifier_branch(branch_value)

    def get_last_state_vector(self):
        # type: (QmlGenericStateCircuitBuilder) -> Optional[sparse.dok_matrix]
        """
        From the last call of :py:func:`build_circuit` the computed (sparse) state vector.

        :return: a sparse vector (shape (n,0)) if present
        """
        return self._last_state_vector
