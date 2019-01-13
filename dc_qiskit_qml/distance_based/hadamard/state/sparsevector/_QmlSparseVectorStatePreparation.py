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
QmlSparseVectorStatePreparation
================================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state.sparsevector._QmlSparseVectorStatePreparation

This is the abstract base class that needs to be implemented for any routine that takes a sparse state vector
and encodes it into a quantum circuit that produces this quantum state.

.. autosummary::
    :nosignatures:

    QmlSparseVectorStatePreparation

QmlSparseVectorStatePreparation
#################################

.. autoclass:: QmlSparseVectorStatePreparation
    :members:

"""
import abc

from qiskit import QuantumCircuit
from scipy import sparse


class QmlSparseVectorStatePreparation(object):
    @abc.abstractmethod
    def prepare_state(self, qc, state_vector):
        # type: (QmlSparseVectorStatePreparation, QuantumCircuit, sparse.dok_matrix) -> QuantumCircuit
        """
        Given a sparse quantum state apply a quantum algorithm (gates) to the given circuit to produce the
        desired quantum state.

        :param qc: the quantum circuit to be used
        :param state_vector: the (complex) state vector of unit length to be prepared
        :return:  the quantum circuit
        """
        pass

    @abc.abstractmethod
    def is_classifier_branch(self, branch_value):
        # type: (QmlSparseVectorStatePreparation, int) -> bool
        """
        The state preparation logic is done in this class, therefore it knows what value the branch measurement
        must have in order to know if we can get classification numbers.

        :param branch_value: the branch measurement value
        :return: True if the branch measurement was 0
        """
        pass