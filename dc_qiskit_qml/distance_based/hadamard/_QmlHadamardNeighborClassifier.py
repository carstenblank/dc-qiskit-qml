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

import logging
import time
from typing import List, Union, Optional, Iterable, Sized

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.measure import measure
from qiskit.providers import BaseBackend, BaseJob, JobStatus
from qiskit.qobj import Qobj
from qiskit.result import Result
from qiskit.result.models import ExperimentResult
from qiskit.transpiler import CouplingMap
from sklearn.base import ClassifierMixin, TransformerMixin

from dc_qiskit_qml.QiskitOptions import QiskitOptions
from .state import QmlStateCircuitBuilder
from ...encoding_maps import EncodingMap

log = logging.getLogger(__name__)


class QmlHadamardNeighborClassifier(ClassifierMixin, TransformerMixin):
    """
    The Hadamard distance & majority based classifier implementing sci-kit learn's mechanism of fit/predict
    """

    def __init__(self, encoding_map, classifier_circuit_factory, backend, shots=1024, coupling_map=None,
                 basis_gates=None, theta=None, options=None):
        # type: (EncodingMap, QmlStateCircuitBuilder, BaseBackend, int, CouplingMap, List[str], Optional[float], Optional[QiskitOptions]) -> None
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
        self.backend = backend  # type: BaseBackend
        self.coupling_map = coupling_map  # type: CouplingMap
        self._X = np.asarray([])  # type: np.ndarray
        self._y = np.asarray([])  # type: np.ndarray
        self.last_predict_X = None
        self.last_predict_label = None
        self.last_predict_probability = []  # type: List[float]
        self._last_predict_circuits = []  # type: List[QuantumCircuit]
        self.last_predict_p_acc = []  # type: List[float]
        self.classifier_state_factory = classifier_circuit_factory  # type: QmlStateCircuitBuilder
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
        # type: (QmlHadamardNeighborClassifier, Iterable) -> None
        """
        Creates the circuits to be executed on the quantum computer

        :param unclassified_input: array like, the input set
        """
        self._last_predict_circuits = []
        if self.classifier_state_factory is None:
            raise Exception("Classifier state factory is not set!")

        for index, x in enumerate(unclassified_input):
            log.info("Creating state for input %d: %s.", index, x)
            circuit_name = 'qml_hadamard_index_%d' % index

            X_input = self.encoding_map.map(x)
            X_train = [self.encoding_map.map(s) for s in self._X]
            qc = self.classifier_state_factory.build_circuit(circuit_name=circuit_name, X_train=X_train,
                                                             y_train=self._y, X_input=X_input)  # type: QuantumCircuit
            ancilla = [q for q in qc.qregs if q.name == 'a'][0]
            qlabel = [q for q in qc.qregs if q.name == 'l^q'][0]
            clabel = [q for q in qc.cregs if q.name == 'l^c'][0]
            branch = [q for q in qc.cregs if q.name == 'b'][0]

            # Classifier
            # Instead of a Hadamard gate we want this to be parametrized
            # use comments for now to toggle!
            # standard.h(qc, ancilla)
            # Must be minus, as the IBMQX gate is implemented this way!
            qc.ry(-self.theta, ancilla)
            qc.z(ancilla)

            # Make sure measurements aren't shifted around
            # This would have some consequences as no gates
            # are allowed after a measurement.
            qc.barrier()

            # The correct label is on ancilla branch |0>!
            measure(qc, ancilla[0], branch[0])
            measure(qc, qlabel, clabel)

            self._last_predict_circuits.append(qc)

    def _read_result(self, test_size, result):
        # type: (QmlHadamardNeighborClassifier, int, Result) -> Optional[List[int]]
        """
        The logic to read out the classification from the result

        :param test_size: the input set size
        :param result: the qiskit result object holding the results from the experiment execution
        :return: if there is a result, will return it as a list of class labels
        """
        self.last_predict_label = []
        self.last_predict_probability = []
        for index in range(test_size):

            # Aggregate Counts
            experiment = None  # type: ExperimentResult
            experiment_names = [experiment.header.name for experiment in result.results
                                if experiment.header and 'qml_hadamard_index_%d' % index in experiment.header.name]
            counts = {}  # type: dict
            for name in experiment_names:
                c = result.get_counts(name)  # type: dict
                for k, v in c.items():
                    if k not in counts:
                        counts[k] = v
                    else:
                        counts[k] += v
            log.debug(counts)
            answer = [
                {'label': int(k.split(' ')[-1], 2), 'branch': int(k.split(' ')[-2], 2), 'count': v}
                for k, v in counts.items()
            ]
            log.info(answer)

            answer_branch = [e for e in answer if self.classifier_state_factory.is_classifier_branch(e['branch'])]
            if len(answer_branch) == 0:
                return None
            p_acc = sum([e['count'] for e in answer_branch]) / self.shots

            sum_of_all = sum([e['count'] for e in answer_branch])
            predicted_answer = max(answer_branch, key=lambda e: e['count'])
            log.debug(predicted_answer)

            predict_label = predicted_answer['label']
            probability = predicted_answer['count'] / sum_of_all

            self.last_predict_label.append(predict_label)
            self.last_predict_probability.append(probability)
            self.last_predict_p_acc.append(p_acc)

        return self.last_predict_label

    def predict_qasm_only(self, X):
        # type: (QmlHadamardNeighborClassifier, Union[Sized, Iterable]) -> Qobj
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
                                         pass_manager=self.options.pass_manager,
                                         callback=self.options.pass_manager)  # type: QuantumCircuit

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
        job = self.backend.run(qobj)  # type: BaseJob
        async_job = AsyncPredictJob(X, job, self)  # type: AsyncPredictJob

        if do_async:
            return async_job

        job.result()
        while not job.status() == JobStatus.DONE:
            if job.status() == JobStatus.CANCELLED:
                break
            log.debug("Waiting for job...")
            time.sleep(10)

        if job.status() == JobStatus.DONE:
            log.info("Circuits Executed!")
            return async_job.predict_result()
        else:
            log.error("Circuits not executed!")
            log.error(job.status)
            return None

    def predict_async(self, X):
        # type: (QmlHadamardNeighborClassifier, any) -> 'AsyncPredictJob'
        """
        Same as predict(self, X, do_aysnc=True)

        :param X: unclassified input set
        :return: Wrapper for a Job
        """
        return self.predict(X, do_async=True)

    def predict_sync(self, X):
        # type: (QmlHadamardNeighborClassifier, any) -> Optional[List[int]]
        """
        Same as predict(self, X, do_aysnc=False)

        :param X: unclassified input set
        :return: List of class labels
        """
        return self.predict(X, do_async=False)

    @staticmethod
    def p_acc_theory(X_train, y_train, X_test):
        # type: (List[np.ndarray], Iterable, np.ndarray) -> float
        """
        Computes the branch acceptance probability

        :param X_train: training set (list of numpy arrays shape (n,1)
        :param y_train: Class labels of training set
        :param X_test: Unclassified input vector (shape (n,1))
        :return: branch acceptance probability P_acc
        """
        M = len(X_train)
        p_acc = sum([np.linalg.norm(X_train[i] + X_test) ** 2 for i, e in enumerate(y_train)]) / (4 * M)
        return p_acc

    @staticmethod
    def p_label_theory(X_train, y_train, X_test, label):
        # type: (List[np.ndarray], Iterable, np.ndarray, int) -> float
        """
        Computes the label's probability

        :param X_train: training set
        :param y_train: Class labels of training set
        :param X_test: Unclassified input vector (shape: (n,1))
        :param label: The label to test
        :return: probability of class label
        """
        M = len(X_train)
        p_acc = QmlHadamardNeighborClassifier.p_acc_theory(X_train, y_train, X_test)
        p_label = sum([np.linalg.norm(X_train[i] + X_test) ** 2
                       for i, e in enumerate(y_train) if e == label]) / (4 * p_acc * M)
        return p_label


class AsyncPredictJob(object):
    """
    Wrapper for a qiskit BaseJob and classification experiments
    """

    def __init__(self, input, job, qml):
        # type: (AsyncPredictJob, Sized, BaseJob, QmlHadamardNeighborClassifier) -> None
        """
        Constructs a new Wrapper

        :param input: the unclassified input data set
        :param job: the qiskit BaseJob running the experiment
        :param qml: the classifier
        """
        self.input = input  # type: Sized
        self.job = job  # type: BaseJob
        self.qml = qml  # type: QmlHadamardNeighborClassifier

    def predict_result(self):
        # type: () -> Optional[List[int]]
        """
        Returns the prediction result if it exists

        :return: a list of class labels or None
        """
        if self.job.status() == JobStatus.DONE:
            log.info("Circuits Executed!")
            return self.qml._read_result(len(self.input), self.job.result())
        else:
            log.error("Circuits not executed!")
            log.error(self.job.status)
            return None
