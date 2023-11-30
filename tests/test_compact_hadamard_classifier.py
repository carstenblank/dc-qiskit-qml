# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import logging
import unittest

import numpy
import qiskit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from dc_qiskit_qml.distance_based.compact_hadamard.compact_hadamard_classifier import CompactHadamardClassifier
from dc_qiskit_qml.encoding_maps import NormedAmplitudeEncoding

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
log = logging.getLogger(__name__)


class FullIris(unittest.TestCase):

    def test(self):
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        X = numpy.asarray([x for x, yy in zip(X, y) if yy != 2])
        y = numpy.asarray([yy for x, yy in zip(X, y) if yy != 2])

        preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('l2norm', Normalizer(norm='l2', copy=True))
        ])
        X = preprocessing_pipeline.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BaseBackend
        qml = CompactHadamardClassifier(backend=execution_backend,
                                        shots=8192,
                                        encoding_map=NormedAmplitudeEncoding())

        qml.fit(X_train, y_train)
        prediction = qml.predict(X_test)
