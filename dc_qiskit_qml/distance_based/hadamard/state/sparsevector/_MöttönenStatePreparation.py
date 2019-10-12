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
MöttönenStatePreparation
==========================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state.sparsevector._MöttönenStatePreparation


.. autosummary::
    :nosignatures:

    MöttönenStatePreparation

MöttönenStatePreparation
#########################

.. autoclass:: MöttönenStatePreparation
    :members:

"""
from dc_qiskit_algorithms.MöttönenStatePreparation import state_prep_möttönen
from qiskit import QuantumCircuit
from scipy import sparse

from ._QmlSparseVectorStatePreparation import QmlSparseVectorStatePreparation


class MöttönenStatePreparation(QmlSparseVectorStatePreparation):
    def prepare_state(self, qc, state_vector):
        # type: (QmlSparseVectorStatePreparation, QuantumCircuit, sparse.dok_matrix) -> QuantumCircuit
        """
        Apply the Möttönen state preparation routine on a quantum circuit using the given state vector.
        The quantum circuit's quantum registers must be able to hold the state vector. Also it is expected
        that the registers are initialized to the ground state ´´|0>´´ each.

        :param qc: the quantum circuit to be used
        :param state_vector: the (complex) state vector of unit length to be prepared
        :return:  the quantum circuit
        """
        qregs = qc.qregs

        # State Prep
        register = [reg[i] for reg in qregs for i in range(0, reg.size)]
        state_prep_möttönen(qc, state_vector, register)

        return qc

    def is_classifier_branch(self, branch_value):
        # type: (QmlSparseVectorStatePreparation, int) -> bool
        """
        The classifier will be directly usable if the branch label is 0.

        :param branch_value: the branch measurement value
        :return: True if the branch measurement was 0
        """
        return branch_value == 0
