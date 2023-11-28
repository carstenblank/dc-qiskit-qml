# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import logging
import unittest

import numpy
import qiskit
from qiskit.providers import BackendV2
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from dc_qiskit_qml.distance_based.hadamard import QmlHadamardNeighborClassifier
from dc_qiskit_qml.distance_based.hadamard.state import QmlBinaryDataStateCircuitBuilder
from dc_qiskit_qml.distance_based.hadamard.state import QmlGenericStateCircuitBuilder
from dc_qiskit_qml.distance_based.hadamard.state.cnot import CCXToffoli
from dc_qiskit_qml.distance_based.hadamard.state.sparsevector import MottonenStatePreparation
from dc_qiskit_qml.distance_based.hadamard.state.sparsevector import FFQRAMStateVectorRoutine
from dc_qiskit_qml.distance_based.hadamard.state.sparsevector import QiskitNativeStatePreparation
from dc_qiskit_qml.encoding_maps import EncodingMap, NormedAmplitudeEncoding

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger(__name__)


def get_data(only_two_features=True):
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.datasets import load_iris

    X, y = load_iris(True)

    X = [x[0:2] if only_two_features else x for x, y in zip(X, y) if y != 2]
    y = [y for x, y in zip(X, y) if y != 2]

    scaler, normalizer = StandardScaler(), Normalizer(norm='l2', copy=True)
    X = scaler.fit_transform(X, y)
    X = normalizer.fit_transform(X, y)

    return X, y


def predict(qml, only_two_features=True):
    # type: (QmlHadamardNeighborClassifier, bool) -> tuple
    X, y = get_data(only_two_features)

    X_train = [X[33], X[85]]
    y_train = [y[33], y[85]]
    X_test = [X[28], X[36]]
    y_test = [y[28], y[36]]

    log.info("Training with %d samples.", len(X_train))

    qml.fit(X_train, y_train)

    log.info("Predict on %d unseen samples.", len(X_test))
    for i in X_test:
        log.info("Predict: %s.", i)

    labels = qml.predict(X_test)

    log.info("Predict: %s (%s%% / %s). Expected: %s" % (labels, qml.last_predict_probability,
                                                        qml.last_predict_p_acc, y_test))

    return labels, y_test


# noinspection NonAsciiCharacters
class QmlHadamardMöttönenTests(unittest.TestCase):

    def runTest(self):
        log.info("Testing 'QmlHadamardNeighborClassifier' with Möttönen Preparation.")
        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2

        classifier_state_factory = QmlGenericStateCircuitBuilder(MottonenStatePreparation())

        qml = QmlHadamardNeighborClassifier(encoding_map=NormedAmplitudeEncoding(),
                                            classifier_circuit_factory=classifier_state_factory,
                                            backend=execution_backend, shots=100 * 8192)
        y_predict, y_test = predict(qml)

        predictions_match = [p == l for p, l in zip(y_predict, y_test)]
        self.assertTrue(all(predictions_match))

        self.assertEqual(len(qml.last_predict_probability), 2)

        input_1_probability = qml.last_predict_probability[0]
        input_2_probability = qml.last_predict_probability[1]

        self.assertAlmostEqual(input_1_probability, 0.629, delta=0.02)
        self.assertAlmostEqual(input_2_probability, 0.547, delta=0.02)


class QmlHadamardFFQramTests(unittest.TestCase):

    def runTest(self):
        log.info("Testing 'QmlHadamardNeighborClassifier' with FF Qram Preparation.")
        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2

        classifier_state_factory = QmlGenericStateCircuitBuilder(FFQRAMStateVectorRoutine())

        qml = QmlHadamardNeighborClassifier(backend=execution_backend,
                                            classifier_circuit_factory=classifier_state_factory,
                                            encoding_map=NormedAmplitudeEncoding(),
                                            shots=100 * 8192)

        y_predict, y_test = predict(qml)

        predictions_match = [p == l for p, l in zip(y_predict, y_test)]
        self.assertTrue(all(predictions_match), "The predictions must be correct.")

        self.assertEqual(len(qml.last_predict_probability), 2)

        input_1_probability = qml.last_predict_probability[0]
        input_2_probability = qml.last_predict_probability[1]

        self.assertAlmostEqual(input_1_probability, 0.629, delta=0.02)
        self.assertAlmostEqual(input_2_probability, 0.547, delta=0.02)


class QmlHadamardQiskitInitializerTests(unittest.TestCase):

    def runTest(self):
        log.info("Testing 'QmlHadamardNeighborClassifier' with FF Qram Preparation.")
        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2

        classifier_state_factory = QmlGenericStateCircuitBuilder(QiskitNativeStatePreparation())

        qml = QmlHadamardNeighborClassifier(backend=execution_backend,
                                            classifier_circuit_factory=classifier_state_factory,
                                            encoding_map=NormedAmplitudeEncoding(),
                                            shots=100 * 8192)

        y_predict, y_test = predict(qml)

        predictions_match = [p == l for p, l in zip(y_predict, y_test)]
        self.assertTrue(all(predictions_match), "The predictions must be correct.")

        self.assertEqual(len(qml.last_predict_probability), 2)

        input_1_probability = qml.last_predict_probability[0]
        input_2_probability = qml.last_predict_probability[1]

        self.assertAlmostEqual(input_1_probability, 0.629, delta=0.02)
        self.assertAlmostEqual(input_2_probability, 0.547, delta=0.02)


# noinspection NonAsciiCharacters
class QmlHadamardMultiClassesTests(unittest.TestCase):

    def runTest(self):
        log.info("Testing 'QmlHadamardNeighborClassifier' with Möttönen Preparation.")
        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2

        classifier_state_factory = QmlGenericStateCircuitBuilder(MottonenStatePreparation())

        qml = QmlHadamardNeighborClassifier(encoding_map=NormedAmplitudeEncoding(),
                                            classifier_circuit_factory=classifier_state_factory,
                                            backend=execution_backend, shots=100 * 8192)

        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler, Normalizer
        from sklearn.pipeline import Pipeline

        X, y = load_wine(True)

        preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca2', PCA(n_components=2)),
            ('l2norm', Normalizer(norm='l2', copy=True))
        ])
        X = preprocessing_pipeline.fit_transform(X, y)

        X_train = X[[33, 88, 144]]
        y_train = y[[33, 88, 144]]

        X_test = X[[28, 140]]
        y_test = y[[28, 140]]

        qml.fit(X_train, y_train)
        prediction = qml.predict(X_test)

        self.assertEqual(len(prediction), len(y_test))
        self.assertListEqual(prediction, list(y_test))


# noinspection NonAsciiCharacters
class QmlHadamardCNOTQubitEncodingTests(unittest.TestCase):

    def runTest(self):
        log.info("Testing 'QmlHadamardNeighborClassifier' with CNOT Preparation.")
        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2

        X_train = numpy.asarray([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        y_train = [0, 1, 0, 1]

        X_test = numpy.asarray([[0.2, 0.4], [0.4, -0.8]])
        y_test = [0, 1]

        class MyEncodingMap(EncodingMap):
            def map(self, input_vector: list) -> sparse.dok_matrix:
                result = sparse.dok_matrix((4, 1))
                index = 0
                if input_vector[0] > 0 and input_vector[1] > 0:
                    index = 0
                if input_vector[0] < 0 < input_vector[1]:
                    index = 1
                if input_vector[0] < 0 and input_vector[1] < 0:
                    index = 2
                if input_vector[0] > 0 > input_vector[1]:
                    index = 3
                result[index, 0] = 1.0
                return result

        encoding_map = MyEncodingMap()

        initial_state_builder = QmlBinaryDataStateCircuitBuilder(CCXToffoli())

        qml = QmlHadamardNeighborClassifier(backend=execution_backend,
                                            shots=100 * 8192,
                                            encoding_map=encoding_map,
                                            classifier_circuit_factory=initial_state_builder)

        qml.fit(X_train, y_train)

        prediction = qml.predict(X_test)

        self.assertEqual(len(prediction), len(y_test))
        self.assertListEqual(prediction, y_test)


class FullIris(unittest.TestCase):

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

        initial_state_builder = QmlGenericStateCircuitBuilder(MottonenStatePreparation())

        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BackendV2
        qml = QmlHadamardNeighborClassifier(backend=execution_backend,
                                            shots=8192,
                                            classifier_circuit_factory=initial_state_builder,
                                            encoding_map=NormedAmplitudeEncoding())

        qml.fit(X_train, y_train)
        prediction = qml.predict(X_test)

        self.assertEqual(len(prediction), len(y_test))
        self.assertListEqual(prediction, list(y_test))

        for i in range(len(qml.last_predict_p_acc)):
            self.assertAlmostEqual(qml.last_predict_p_acc[i],
                                   QmlHadamardNeighborClassifier.p_acc_theory(X_train, y_train, X_test[i]), delta=0.05)

        for i in range(len(qml.last_predict_probability)):
            predicted_label = prediction[i]
            self.assertAlmostEqual(qml.last_predict_probability[i],
                                   QmlHadamardNeighborClassifier.p_label_theory(X_train, y_train, X_test[i], predicted_label),
                                   delta=0.05)
