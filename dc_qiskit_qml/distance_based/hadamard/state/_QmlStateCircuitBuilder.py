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
QmlStateCircuitBuilder
==============================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state._QmlStateCircuitBuilder

This is the abstract base class to implement a custom state circuit builder which takes the classification
training data samples and labels and one to be classified sample and outputs the circuit that creates the necessary
quantum state than then can be used to apply the :py:class:`QmlHadamardNeighborClassifier`.

.. autosummary::
    :nosignatures:

    QmlStateCircuitBuilder

QmlStateCircuitBuilder
##############################

.. autoclass:: QmlStateCircuitBuilder
    :members:

"""
import abc
from typing import List

from qiskit import QuantumCircuit
from scipy import sparse


class QmlStateCircuitBuilder(object):
    """
    Interface class for creating a quantum circuit from a sparse quantum state vector.
    """

    @abc.abstractmethod
    def build_circuit(self, circuit_name, X_train, y_train, X_input):
        # type: (QmlStateCircuitBuilder, str, List[sparse.dok_matrix], any, sparse.dok_matrix) -> QuantumCircuit
        """
        Build a circuit that encodes the training (samples/labels) and input data sets into a quantum circuit

        :param circuit_name: The name of the quantum circuit
        :param X_train: The training data set
        :param y_train: the training class label data set
        :param X_input: the unclassified input data vector
        :return: The circuit containing the gates to encode the input data
        """
        pass

    @abc.abstractmethod
    def is_classifier_branch(self, branch_value):
        # type: (QmlStateCircuitBuilder, int) -> bool
        """
        As each state preparation algorithm uses a unique layout. The classifier has the correct classification
        probabilities only on the correct branch of the ancilla qubit. However each state preparation may have
        different methods making it necessary to query specific logic to assess what value must be given after the
        branch qubit measurement.

        :param branch_value: The measurement of the branch
        :return: True is the branch is containing the classification, False if not
        """
        pass

