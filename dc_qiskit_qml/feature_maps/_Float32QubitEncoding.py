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

from typing import List

import bitstring
import numpy as np
from scipy import sparse

from . import FeatureMap


class Float32QubitEncoding(FeatureMap):
    def map(self, x):
        # type: (Float32QubitEncoding, List[complex]) -> sparse.dok_matrix
        x_array = np.asarray(x)
        x_norm = np.sqrt(x_array.size)
        feature_length = x_array.shape[0] * 32
        feature_x = sparse.dok_matrix((2**feature_length, 1), dtype=complex)  # type: sparse.dok_matrix
        e = None  # type: np.float64
        for e in x_array:
            sbit = bitstring.pack('float:32', e)
            qubit_state = sbit.bin
            index_for = int(qubit_state, 2)
            feature_x[index_for, 0] = 1.0/x_norm
        return feature_x