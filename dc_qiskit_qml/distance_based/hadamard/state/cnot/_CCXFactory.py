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
CCXFactory
============

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state.cnot._CCXFactory

This class is the abstract base class for the multiple-controlled X-gates (short ccx) that are primarily used
for binary input data of the classifier.

.. autosummary::
    :nosignatures:

    CCXFactory

CCXFactory
#############

.. autoclass:: CCXFactory
    :members:

"""

import abc
from typing import List, Union, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.register import Register


class CCXFactory(object):
    """
    Abstract base class for using a multi-ccx gate within the context of the classifier
    """
    @abc.abstractmethod
    def ccx(self, qc, conditial_case, control_qubits, tgt):
        # type: (CCXFactory, QuantumCircuit, int, Union[List[Tuple[Register, int]], QuantumRegister], Union[Tuple[Register, int], QuantumRegister]) ->QuantumCircuit
        """
        This method applies to a quantum circuit on the control qubits the desired controlled operation on the target
        if the quantum state coincides with the (binary representation of the) conditional case.

        This abstract method must be implemented in order to be used by the
        :py:class:`_QmlUniformAmplitudeStateCircuitBuilder` and so that the correct state for the classifier can be
        applied.

        :param qc: the circuit to apply this operation to
        :param conditial_case: an integer whose binary representation signifies the branch to controll on
        :param control_qubits: the qubits that hold the desired conditional case
        :param tgt: the target to be applied the controlled X gate on
        :return: the circuit after application of the gate
        """
        pass