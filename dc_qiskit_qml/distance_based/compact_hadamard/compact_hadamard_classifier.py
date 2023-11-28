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
QmlHadamardNeighborClassifier
===============================

.. currentmodule:: dc_qiskit_qml.distance_based.hadamard._QmlHadamardNeighborClassifier

Implementing the Hadamard distance & majority based classifier (http://stacks.iop.org/0295-5075/119/i=6/a=60002).

.. autosummary::
   :nosignatures:

   QmlHadamardNeighborClassifier
   AsyncPredictJob

More details:

QmlHadamardNeighborClassifier
###############################

.. autoclass:: QmlHadamardNeighborClassifier
    :members:

AsyncPredictJob
##################

.. autoclass:: AsyncPredictJob
    :members:

"""
import itertools
import logging
from typing import List, Union, Optional, Iterable, Sized

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2, JobV1
from qiskit.qobj import QasmQobj
from qiskit.transpiler import CouplingMap
from sklearn.base import ClassifierMixin, TransformerMixin

from dc_qiskit_qml.QiskitOptions import QiskitOptions
from ...encoding_maps import EncodingMap

log = logging.getLogger(__name__)


class CompactHadamardClassifier(ClassifierMixin, TransformerMixin):
    """
    The Hadamard distance & majority based classifier implementing sci-kit learn's mechanism of fit/predict
    """

    def __init__(self, encoding_map, backend, shots=1024, coupling_map=None,
                 basis_gates=None, theta=None, options=None):
        # type: (EncodingMap, BackendV2, int, CouplingMap, List[str], Optional[float], Optional[QiskitOptions]) -> None
        """
        Create the classifier

        :param encoding_map: a classical feature map to apply to every training and testing sample before building
        the circuit
        :param classifier_circuit_factory: the circuit building factory class
        :param backend: the qiskit backend to do the compilation & computation on
        :param shots: *deprecated* use options. the amount of shots for the experiment
        :param coupling_map: *deprecated* use options. if given overrides the backend's coupling map, useful when using the simulator
        :param basis_gates: *deprecated* use options. if given overrides the backend's basis gates, useful for the simulator
        :param theta: an advanced feature that generalizes the "Hadamard" gate as a rotation with this angle
        :param options: the options for transpilation & executions with qiskit.
        """
        self.encoding_map = encoding_map  # type: EncodingMap
        self.basis_gates = basis_gates  # type: List[str]
        self.shots = shots  # type: int
        self.backend = backend  # type: BackendV2
        self.coupling_map = coupling_map  # type: CouplingMap
        self._X = np.asarray([])  # type: np.ndarray
        self._y = np.asarray([])  # type: np.ndarray
        self.last_predict_X = None
        self.last_predict_label = None
        self.last_predict_probability = []  # type: List[float]
        self._last_predict_circuits = []  # type: List[QuantumCircuit]
        self.last_predict_p_acc = []  # type: List[float]
        self.theta = np.pi / 2 if theta is None else theta  # type: float

        if options is not None:
            self.options = options  # type: QiskitOptions
        else:
            self.options = QiskitOptions()

        self.options.basis_gates = basis_gates
        self.options.coupling_map = coupling_map
        self.options.shots = shots

    def transform(self, X, y='deprecated', copy=None):
        return X

    def fit(self, X, y):
        # type: (QmlHadamardNeighborClassifier, Iterable) -> QmlHadamardNeighborClassifier
        """
        Internal fit method just saves the train sample set

        :param X: array_like, training sample
        """
        self._X = np.asarray(X)
        self._y = y
        log.debug("Setting training data:")
        for x, y in zip(self._X, self._y):
            log.debug("%s: %s", x, y)

        return self

    def _create_circuits(self, unclassified_input):
        # type: (Iterable) -> None
        """
        Creates the circuits to be executed on the quantum computer

        :param unclassified_input: array like, the input set
        """
        self._last_predict_circuits = []

        for index, x in enumerate(unclassified_input):
            log.info("Creating state for input %d: %s.", index, x)
            circuit_name = 'qml_hadamard_index_%d' % index

            X_input = self.encoding_map.map(x)
            X_train_0 = [self.encoding_map.map(s) for s, l in zip(self._X, self._y) if l == 0]
            X_train_1 = [self.encoding_map.map(s) for s, l in zip(self._X, self._y) if l == 1]

            full_data = list(itertools.zip_longest([X_train_0, X_train_1], fillvalue=None))

            # all_batches = max(len(X_train_0), len(X_train_1))
            #
            # index_no =
            #
            # ancilla = QuantumRegister(ancilla_qubits_needed, "a")
            # index = QuantumRegister(index_of_samples_qubits_needed, "i")
            # data = QuantumRegister(sample_space_dimensions_qubits_needed, "f^S")
            # clabel = ClassicalRegister(label_qubits_needed, "l^c")

            qc = QuantumCircuit()
            self._last_predict_circuits.append(qc)

    def predict_qasm_only(self, X):
        # type: (Union[Sized, Iterable]) -> QasmQobj
        """
        Instead of predicting straight away returns the Qobj, the command object for executing
        the experiment

        :param X: array like, unclassified input set
        :return: the compiled Qobj ready for execution!
        """
        self.last_predict_X = X
        self.last_predict_label = []
        self._last_predict_circuits = []
        self.last_predict_probability = []
        self.last_predict_p_acc = []
        log.info("Creating circuits (#%d inputs)..." % len(X))
        self._create_circuits(X)

        log.info("Compiling circuits...")

        transpiled_qc = qiskit.transpile(self._last_predict_circuits,
                                         backend=self.backend,
                                         coupling_map=self.options.coupling_map,
                                         basis_gates=self.options.basis_gates,
                                         backend_properties=self.options.backend_properties,
                                         initial_layout=self.options.initial_layout,
                                         seed_transpiler=self.options.seed_transpiler,
                                         optimization_level=self.options.optimization_level,
                                         callback=self.options.pass_manager)  # type: List[QuantumCircuit]

        qobj = qiskit.assemble(transpiled_qc,
                               backend=self.backend,
                               shots=self.options.shots,
                               qobj_id=self.options.qobj_id,
                               qobj_header=self.options.qobj_header,
                               memory=self.options.memory,
                               max_credits=self.options.max_credits,
                               seed_simulator=self.options.seed_simulator,
                               default_qubit_los=self.options.default_qubit_los,
                               default_meas_los=self.options.default_meas_los,
                               schedule_los=self.options.schedule_los,
                               meas_level=self.options.meas_level,
                               meas_return=self.options.meas_return,
                               memory_slots=self.options.memory_slots,
                               memory_slot_size=self.options.memory_slot_size,
                               rep_time=self.options.rep_time,
                               parameter_binds=self.options.parameter_binds,
                               **self.options.run_config
                               )

        return qobj

    def predict(self, X, do_async=False):
        # type: (QmlHadamardNeighborClassifier, Union[Sized, Iterable], bool) -> Union[Optional[List[int]], 'AsyncPredictJob']
        """
        Predict the class labels of the unclassified input set!

        :param X: array like, the unclassified input set
        :param do_async: if True return a wrapped job, it is handy for reading out the prediction results from a real processor
        :return: depending on the input either the prediction on class labels or a wrapper object for an async task
        """
        qobj = self.predict_qasm_only(X)
        log.info("Executing circuits...")
        job = self.backend.run(qobj)  # type: JobV1
