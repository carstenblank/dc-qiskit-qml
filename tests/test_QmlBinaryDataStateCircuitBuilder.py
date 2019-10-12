# -*- coding: utf-8 -*-

# Copyright 2018, Carsten Blank.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import logging
import sys
import unittest

import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.measure import measure
from qiskit.providers import BaseBackend, BaseJob
from scipy import sparse

from dc_qiskit_qml.distance_based.hadamard.state import QmlBinaryDataStateCircuitBuilder
from dc_qiskit_qml.distance_based.hadamard.state.cnot import CCXMöttönen, CCXToffoli
from dc_qiskit_qml.feature_maps import FeatureMap
from dc_qiskit_qml.feature_maps import FixedLengthQubitEncoding

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(logging._defaultFormatter)
logger.addHandler(stream_handler)


def extract_gate_info(qc, index):
    # type: (QuantumCircuit, int) -> list
    return [qc.data[index][0].name, qc.data[index][0].params, str(qc.data[index][1][-1])]


class QubitEncodingClassifierStateCircuitTests(unittest.TestCase):

    def test_one(self):
        feature_map = FixedLengthQubitEncoding(4, 4)

        X_train = [
            [4.4, -9.53],
            [18.42, 1.0]
        ]
        y_train = [0, 1]
        input = [2.043, 13.84]

        X_train_in_feature_space = [feature_map.map(x) for x in X_train]
        X_train_in_feature_space_qubit_notation = [["{:b}".format(i).zfill(2 * 9) for i, _ in elem.keys()] for elem in
                                                   X_train_in_feature_space]

        input_in_feature_space = feature_map.map(input)
        input_in_feature_space_qubit_notation = ["{:b}".format(i).zfill(2 * 9) for i, _ in
                                                 input_in_feature_space.keys()]

        logger.info("Training samples in feature space: {}".format(X_train_in_feature_space_qubit_notation))
        logger.info("Input sample in feature space: {}".format(input_in_feature_space_qubit_notation))

        circuit = QmlBinaryDataStateCircuitBuilder(CCXMöttönen())

        qc = circuit.build_circuit('test', X_train=X_train_in_feature_space, y_train=y_train, X_input=input_in_feature_space)

        self.assertIsNotNone(qc)
        self.assertIsNotNone(qc.data)
        self.assertEqual(30, len(qc.data))

        self.assertListEqual(["h"], [qc.data[0][0].name])
        self.assertListEqual(["h"], [qc.data[1][0].name])

        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 2))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 3))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 4))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 8)"], extract_gate_info(qc, 5))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 10)"], extract_gate_info(qc, 6))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 11)"], extract_gate_info(qc, 7))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(18, 'f^S'), 15)"], extract_gate_info(qc, 8))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 0)"], extract_gate_info(qc, 9))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 2)"], extract_gate_info(qc, 10))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 11))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 12))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 6)"], extract_gate_info(qc, 13))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 14))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(18, 'f^S'), 14)"], extract_gate_info(qc, 15))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 16))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(18, 'f^S'), 10)"], extract_gate_info(qc, 17))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(18, 'f^S'), 11)"], extract_gate_info(qc, 18))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(18, 'f^S'), 13)"], extract_gate_info(qc, 19))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(18, 'f^S'), 16)"], extract_gate_info(qc, 20))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 21))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 0)"], extract_gate_info(qc, 22))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 2)"], extract_gate_info(qc, 23))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 24))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 25))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 6)"], extract_gate_info(qc, 26))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 27))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(18, 'f^S'), 14)"], extract_gate_info(qc, 28))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 29))

    def test_two(self):

        feature_map = FixedLengthQubitEncoding(2, 2)

        X_train = [
            [4.4, -9.53],
            [18.42, 1.0]
        ]
        y_train = [0, 1]
        input = [2.043, 13.84]

        X_train_in_feature_space = [feature_map.map(x) for x in X_train]
        X_train_in_feature_space_qubit_notation = [["{:b}".format(i).zfill(2*5) for i, _ in elem.keys()] for elem in X_train_in_feature_space]

        input_in_feature_space = feature_map.map(input)
        input_in_feature_space_qubit_notation = ["{:b}".format(i).zfill(2*5) for i, _ in input_in_feature_space.keys()]

        logger.info("Training samples in feature space: {}".format(X_train_in_feature_space_qubit_notation))
        logger.info("Input sample in feature space: {}".format(input_in_feature_space_qubit_notation))

        circuit = QmlBinaryDataStateCircuitBuilder(CCXMöttönen())

        qc = circuit.build_circuit('test', X_train=X_train_in_feature_space, y_train=y_train, X_input=input_in_feature_space)

        self.assertIsNotNone(qc)
        self.assertIsNotNone(qc.data)
        self.assertEqual(len(qc.data), 22)

        self.assertListEqual(["h"], [qc.data[0][0].name])
        self.assertListEqual(["h"], [qc.data[1][0].name])

        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 2))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 3))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(10, 'f^S'), 4)"], extract_gate_info(qc, 4))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(10, 'f^S'), 5)"], extract_gate_info(qc, 5))
        self.assertListEqual(["ccx_uni_rot", [0], "Qubit(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 6))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(10, 'f^S'), 0)"], extract_gate_info(qc, 7))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 8))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 9))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 10))
        self.assertListEqual(["ccx_uni_rot", [1], "Qubit(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 11))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 12))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(10, 'f^S'), 5)"], extract_gate_info(qc, 13))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 14))
        self.assertListEqual(["ccx_uni_rot", [2], "Qubit(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 15))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(10, 'f^S'), 0)"], extract_gate_info(qc, 16))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 17))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 18))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 19))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 20))
        self.assertListEqual(["ccx_uni_rot", [3], "Qubit(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 21))

        qregs = qc.qregs
        cregs = [ClassicalRegister(qr.size, 'c' + qr.name) for qr in qregs]
        qc2 = QuantumCircuit(*qregs, *cregs, name='test2')

        qc2.data = qc.data
        for i in range(len(qregs)):
            measure(qc2, qregs[i], cregs[i])

        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BaseBackend
        qobj = qiskit.compile([qc2], execution_backend, shots=8192)
        result = execution_backend.run(qobj)  # type: BaseJob
        counts = result.result().get_counts()  # type: dict

        self.assertListEqual(sorted(counts.keys()), sorted(['0 0100111010 0 0', '0 0100001111 0 1', '1 0100100100 1 0', '1 0100001111 1 1']))

    def test_three(self):
        feature_map = FixedLengthQubitEncoding(4, 4)

        X_train = [
            [4.4, -9.53],
            [18.42, 1.0]
        ]
        y_train = [0, 1]
        input = [2.043, 13.84]

        X_train_in_feature_space = [feature_map.map(x) for x in X_train]
        X_train_in_feature_space_qubit_notation = [["{:b}".format(i).zfill(2 * 9) for i, _ in elem.keys()] for elem in
                                                   X_train_in_feature_space]

        input_in_feature_space = feature_map.map(input)
        input_in_feature_space_qubit_notation = ["{:b}".format(i).zfill(2 * 9) for i, _ in
                                                 input_in_feature_space.keys()]

        logger.info("Training samples in feature space: {}".format(X_train_in_feature_space_qubit_notation))
        logger.info("Input sample in feature space: {}".format(input_in_feature_space_qubit_notation))

        circuit = QmlBinaryDataStateCircuitBuilder(CCXToffoli())

        qc = circuit.build_circuit('test', X_train=X_train_in_feature_space, y_train=y_train, X_input=input_in_feature_space)

        self.assertIsNotNone(qc)
        self.assertIsNotNone(qc.data)
        # TODO: adjust ASAP
        # self.assertEqual(36, len(qc.data))
        #
        # self.assertListEqual(["h"], [qc.data[0].name])
        # self.assertListEqual(["h"], [qc.data[1].name])
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 2))
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'i'), 0)"], extract_gate_info(qc, 3))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 4))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 5))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 6))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 8)"], extract_gate_info(qc, 7))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 10)"], extract_gate_info(qc, 8))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 11)"], extract_gate_info(qc, 9))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 15)"], extract_gate_info(qc, 10))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 11))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 0)"], extract_gate_info(qc, 12))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 2)"], extract_gate_info(qc, 13))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 14))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 15))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 6)"], extract_gate_info(qc, 16))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 17))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 14)"], extract_gate_info(qc, 18))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'i'), 0)"], extract_gate_info(qc, 19))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 20))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 21))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 10)"], extract_gate_info(qc, 22))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 11)"], extract_gate_info(qc, 23))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 13)"], extract_gate_info(qc, 24))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 16)"], extract_gate_info(qc, 25))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 26))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 27))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 0)"], extract_gate_info(qc, 28))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 2)"], extract_gate_info(qc, 29))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 3)"], extract_gate_info(qc, 30))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 4)"], extract_gate_info(qc, 31))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 6)"], extract_gate_info(qc, 32))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 7)"], extract_gate_info(qc, 33))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(18, 'f^S'), 14)"], extract_gate_info(qc, 34))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 35))

    def test_four(self):

        feature_map = FixedLengthQubitEncoding(2, 2)

        X_train = [
            [4.4, -9.53],
            [18.42, 1.0]
        ]
        y_train = [0, 1]
        input = [2.043, 13.84]

        X_train_in_feature_space = [feature_map.map(x) for x in X_train]
        X_train_in_feature_space_qubit_notation = [["{:b}".format(i).zfill(2*5) for i, _ in elem.keys()] for elem in X_train_in_feature_space]

        input_in_feature_space = feature_map.map(input)
        input_in_feature_space_qubit_notation = ["{:b}".format(i).zfill(2*5) for i, _ in input_in_feature_space.keys()]

        logger.info("Training samples in feature space: {}".format(X_train_in_feature_space_qubit_notation))
        logger.info("Input sample in feature space: {}".format(input_in_feature_space_qubit_notation))

        circuit = QmlBinaryDataStateCircuitBuilder(CCXToffoli())
        qc = circuit.build_circuit('test', X_train=X_train_in_feature_space, y_train=y_train, X_input=input_in_feature_space)

        self.assertIsNotNone(qc)
        self.assertIsNotNone(qc.data)
        # TODO: adjust ASAP
        # self.assertEqual(28, len(qc.data))
        #
        # self.assertListEqual(["h"], [qc.data[0].name])
        # self.assertListEqual(["h"], [qc.data[1].name])
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 2))
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'i'), 0)"], extract_gate_info(qc, 3))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 4))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 5))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 4)"], extract_gate_info(qc, 6))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 5)"], extract_gate_info(qc, 7))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 8))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 9))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 0)"], extract_gate_info(qc, 10))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 11))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 12))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 13))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 14))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'i'), 0)"], extract_gate_info(qc, 15))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 16))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 17))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 5)"], extract_gate_info(qc, 18))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 19))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 20))
        #
        # self.assertListEqual(["x", [], "(QuantumRegister(1, 'a'), 0)"], extract_gate_info(qc, 21))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 0)"], extract_gate_info(qc, 22))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 1)"], extract_gate_info(qc, 23))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 2)"], extract_gate_info(qc, 24))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 3)"], extract_gate_info(qc, 25))
        # self.assertListEqual(["ccx", [], "(QuantumRegister(10, 'f^S'), 8)"], extract_gate_info(qc, 26))
        #
        # self.assertListEqual(["ccx", [], "(QuantumRegister(1, 'l^q'), 0)"], extract_gate_info(qc, 27))

        cregs = [ClassicalRegister(qr.size, 'c' + qr.name) for qr in qc.qregs]
        qc2 = QuantumCircuit(*qc.qregs, *cregs, name='test2')

        qc2.data = qc.data
        for i in range(len(qc.qregs)):
            measure(qc2, qc.qregs[i], cregs[i])

        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BaseBackend
        qobj = qiskit.compile([qc2], execution_backend, shots=8192)
        result = execution_backend.run(qobj)  # type: BaseJob
        counts = result.result().get_counts()  # type: dict

        self.assertListEqual(sorted(['0 0100111010 0 0', '0 0100001111 0 1', '1 0100100100 1 0', '1 0100001111 1 1']), sorted(counts.keys()))

    def test_five(self):
        X_train = np.asarray([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        y_train = [0, 1, 0, 1]

        X_test = np.asarray([[0.2, 0.4], [0.4, -0.8]])
        y_test = [0, 1]

        class MyFeatureMap(FeatureMap):
            def map(self, input_vector: list) -> sparse.dok_matrix:
                result = sparse.dok_matrix((4, 1))
                index = 0
                if input_vector[0] > 0 and input_vector[1] > 0:
                    index = 0
                if input_vector[0] < 0 and input_vector[1] > 0:
                    index = 1
                if input_vector[0] < 0 and input_vector[1] < 0:
                    index = 2
                if input_vector[0] > 0 and input_vector[1] < 0:
                    index = 3
                result[index, 0] = 1.0
                return result

        initial_state_builder = QmlBinaryDataStateCircuitBuilder(CCXToffoli())

        feature_map = MyFeatureMap()
        X_train_in_feature_space = [feature_map.map(s) for s in X_train]
        X_test_in_feature_space = [feature_map.map(s) for s in X_test]
        qc = initial_state_builder.build_circuit('test', X_train_in_feature_space, y_train, X_test_in_feature_space[0])

        self.assertIsNotNone(qc)
        self.assertIsNotNone(qc.data)
        # self.assertEqual(len(qc.data), 28)

        # self.assertListEqual(["h"], [qc.data[0].name])
        # self.assertListEqual(["h"], [qc.data[1].name])

        qregs = qc.qregs
        cregs = [ClassicalRegister(qr.size, 'c_' + qr.name) for qr in qregs]
        qc2 = QuantumCircuit(*qregs, *cregs, name='test2')

        qc2.data = qc.data
        for i in range(len(qregs)):
            measure(qc2, qregs[i], cregs[i])

        execution_backend = qiskit.Aer.get_backend('qasm_simulator')  # type: BaseBackend
        qobj = qiskit.compile([qc2], execution_backend, shots=8192)
        result = execution_backend.run(qobj)  # type. BaseJob
        counts = result.result().get_counts()  # type: dict

        self.assertListEqual(
            sorted([
                '00 0 00 00 0',
                '00 0 00 00 1',
                '00 1 01 01 0',
                '00 1 00 01 1',
                '00 0 10 10 0',
                '00 0 00 10 1',
                '00 1 11 11 0',
                '00 1 00 11 1'
            ]),
            sorted(counts.keys())
        )

