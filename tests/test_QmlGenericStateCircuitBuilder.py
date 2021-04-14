import logging
import unittest

import numpy
import qiskit
from qiskit.providers import BaseBackend, BaseJob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from dc_qiskit_qml.distance_based.hadamard.state import QmlGenericStateCircuitBuilder
from dc_qiskit_qml.distance_based.hadamard.state.sparsevector import MöttönenStatePreparation
from dc_qiskit_qml.encoding_maps import IdentityEncodingMap

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger(__name__)


class MöttönenStatePreparationTest(unittest.TestCase):

    def runTest(self):
        from sklearn.datasets import load_iris

        X, y = load_iris(True)
        X = numpy.asarray([x[0:2] for x, yy in zip(X, y) if yy != 2])
        y = numpy.asarray([yy for x, yy in zip(X, y) if yy != 2])

        preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('l2norm', Normalizer(norm='l2', copy=True))
        ])
        X = preprocessing_pipeline.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        encoding_map = IdentityEncodingMap()
        initial_state_builder = QmlGenericStateCircuitBuilder(MöttönenStatePreparation())

        qc = initial_state_builder.build_circuit("test", [encoding_map.map(e) for e in X_train], y_train,
                                                 encoding_map.map(X_test[0]))
        statevector_backend = qiskit.Aer.get_backend('statevector_simulator')  # type: BaseBackend
        job = qiskit.execute(qc, statevector_backend, shots=1)  # type: BaseJob
        simulator_state_vector = job.result().get_statevector()
        input_state_vector = initial_state_builder.get_last_state_vector()

        phase = set(numpy.angle(simulator_state_vector)[numpy.abs(simulator_state_vector) > 1e-3]).pop()
        simulator_state_vector[numpy.abs(simulator_state_vector) < 1e-3] = 0
        simulator_state_vector = numpy.exp(-1.0j * phase) * simulator_state_vector
        for i, s in zip(input_state_vector.toarray(), simulator_state_vector):
            log.debug("{:.4f} == {:.4f}".format(i[0], s))
            self.assertAlmostEqual(i[0], s, places=3)
