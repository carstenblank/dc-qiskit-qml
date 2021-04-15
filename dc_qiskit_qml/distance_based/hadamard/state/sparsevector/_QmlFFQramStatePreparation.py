# -*- coding: utf-8 -*-
from typing import List

import numpy as np
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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.measure import measure
from scipy import sparse

from dc_qiskit_algorithms.FlipFlopQuantumRam import FFQramDb, add_vector
from ._QmlSparseVectorStatePreparation import QmlSparseVectorStatePreparation


class FFQRAMStateVectorRoutine(QmlSparseVectorStatePreparation):
    def prepare_state(self, qc, state_vector):
        # type: (FFQRAMStateVectorRoutine, QuantumCircuit, sparse.dok_matrix) -> QuantumCircuit

        cregs_new = [reg for reg in qc.cregs if reg.name != "b"]  # type: List[ClassicalRegister]
        branch_old = [reg for reg in qc.cregs if reg.name == "b"][0]  # type: ClassicalRegister
        branch = ClassicalRegister(name=branch_old.name, size=2)
        ffqram_reg = QuantumRegister(1, "ffqram_reg")

        qc_result = QuantumCircuit(*qc.qregs, ffqram_reg, *cregs_new, branch, name=qc.name)
        bus = [reg[i] for reg in qc.qregs for i in range(reg.size)]

        # Create DB
        db = FFQramDb()
        # TODO: make it take a sparse matrix too!
        add_vector(db, list([np.real(e[0]) for e in state_vector.toarray()]))

        # State Prep
        for r in bus:
            qc_result.h(r)
        db.add_to_circuit(qc_result, bus, ffqram_reg[0])

        qc_result.barrier()
        measure(qc_result, ffqram_reg[0], branch[1])

        return qc_result

    def is_classifier_branch(self, branch_value):
        return branch_value == 2
