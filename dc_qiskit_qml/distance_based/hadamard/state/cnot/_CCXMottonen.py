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
CCXMöttönen
============

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state.cnot._CCXMöttönen

This module implements a :py:class:`CCXFactory` to create a multi-controlled X-gate (or NOT gate) to a circuit
using the uniform rotations as described by Möttönen et al. (http://dl.acm.org/citation.cfm?id=2011670.2011675).

.. autosummary::
    :nosignatures:

    CCXMöttönen

CCXMöttönen
#############

.. autoclass:: CCXMöttönen
    :members:

"""
import logging
from typing import List, Union, Tuple

from qclib.gates.mcx import LinearMcx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.register import Register

from . import CCXFactory

log = logging.getLogger(__name__)


class CCXMottonen(CCXFactory):
    """
    cc-X gate implemented via the uniform rotation scheme (Möttönen et al. 2005)
    """
    def ccx(self, qc, conditial_case, control_qubits, tgt):
        # type: (CCXFactory, QuantumCircuit, int, Union[List[Tuple[Register, int]], QuantumRegister], Union[Tuple[Register, int], QuantumRegister]) ->QuantumCircuit
        """
        Using the Möttönen uniform rotation, one can create multi-controlled NOT gate
        (along with other arbitrary rotations). The routine has exponential (w.r.t. number of qubits) gate usage.

        :param qc: the circuit to apply this operation to
        :param conditial_case: an integer whose binary representation signifies the branch to controll on
        :param control_qubits: the qubits that hold the desired conditional case
        :param tgt: the target to be applied the controlled X gate on
        :return: the circuit after application of the gate
        """
        LinearMcx.mcx(qc, ctrl_state=conditial_case, controls=control_qubits, target=tgt)
        return qc
