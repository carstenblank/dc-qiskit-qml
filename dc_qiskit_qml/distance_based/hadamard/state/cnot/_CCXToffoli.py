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
CCXToffoli
============

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard.state.cnot._CCXToffoli

This module implements a :py:class:`CCXFactory` to create a multi-controlled X-gate (or NOT gate) to a circuit
using the Toffoli gate cascade on ancillary qubits described by Nielsen & Chuang (https://doi.org/10.1017/CBO9780511976667).
Its advantage is that it doesn't need expnential gates w.r.t. to the controlled qubit count, however, its downside is
that it needs auxiliary qubits. It depends strongly on the use case whether to use this or the algorithm by
Möttönen et al. as implemented by :py:class:`CCXMöttönen`.

.. autosummary::
    :nosignatures:

    CCXToffoli

CCXToffoli
#############

.. autoclass:: CCXToffoli
    :members:

"""

import logging
from typing import List, Union, Tuple

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.register import Register
from qiskit.extensions.standard.barrier import barrier
from qiskit.extensions.standard.ccx import ccx
from qiskit.extensions.standard.cx import cx
from qiskit.extensions.standard.x import x

from . import CCXFactory

log = logging.getLogger(__name__)


class CCXToffoli(CCXFactory):
    """
    cc-X gate implemented via the Toffoli cascade with auxiliary qubits.
    """
    def ccx(self, qc, conditial_case, control_qubits, tgt):
        # type:(CCXToffoli, QuantumCircuit, int, List[Tuple[Register, int]], Union[Tuple[Register, int], QuantumRegister]) -> QuantumCircuit
        """
        Using the Toffoli gate cascade on auxiliary qubits, one can create multi-controlled NOT gate. The routine
        needs to add an auxiliary register called ``
        ccx_ancilla`` which also will be re-used if multiple calls
        of this routine are done within the same circuit. As the cascade is always 'undone', the ancilla register
        is left to the ground state.

        :param qc: the circuit to apply this operation to
        :param conditial_case: an integer whose binary representation signifies the branch to controll on
        :param control_qubits: the qubits that hold the desired conditional case
        :param tgt: the target to be applied the controlled X gate on
        :return: the circuit after application of the gate
        """

        # Prepare conditional case
        bit_string = "{:b}".format(conditial_case).zfill(len(control_qubits))
        for i, b in enumerate(reversed(bit_string)):
            if b == '0':
                x(qc, control_qubits[i])
        barrier(qc)

        if len(control_qubits) == 1: # This is just the normal CNOT
            cx(qc, control_qubits[0], tgt)
        elif len(control_qubits) == 2: # This is the simple Toffoli
            ccx(qc, control_qubits[0], control_qubits[1], tgt)
        else:
            # Create ancilla qubit or take the one that is already there
            if 'ccx_ancilla' not in [q.name for q in qc.qregs]:
                ccx_ancilla = QuantumRegister(len(control_qubits) - 1, 'ccx_ancilla')  # type: QuantumRegister
                qc.add_register(ccx_ancilla)
            else:
                ccx_ancilla = [q for q in qc.qregs if q.name == 'ccx_ancilla'][0]  # type: QuantumRegister

            # Algorithm
            ccx(qc, control_qubits[0], control_qubits[1], ccx_ancilla[0])
            for i in range(1, ccx_ancilla.size):
                ccx(qc, control_qubits[i], ccx_ancilla[i - 1], ccx_ancilla[i])

            ccx(qc, control_qubits[-1], ccx_ancilla[ccx_ancilla.size - 1], tgt)

            for i in reversed(range(1, ccx_ancilla.size)):
                ccx(qc, control_qubits[i], ccx_ancilla[i - 1], ccx_ancilla[i])
            ccx(qc, control_qubits[0], control_qubits[1], ccx_ancilla[0])

        barrier(qc)
        # Undo the conditional case
        for i, b in enumerate(reversed(bit_string)):
            if b == '0':
                x(qc, control_qubits[i])

        return qc
