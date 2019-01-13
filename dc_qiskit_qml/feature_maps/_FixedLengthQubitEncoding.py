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

import numpy as np
from scipy import sparse

from . import FeatureMap


class FixedLengthQubitEncoding(FeatureMap):
    def __init__(self, integer_length, decimal_length):
        self.integer_length = integer_length
        self.decimal_length = decimal_length

    def map(self, x):
        # type: (FixedLengthQubitEncoding, List[complex]) -> sparse.dok_matrix
        x_array = np.asarray(x)
        feature_length = x_array.shape[0] * (1 + self.integer_length + self.decimal_length)
        feature_x = sparse.dok_matrix((2**feature_length, 1), dtype=complex)  # type: sparse.dok_matrix
        e = None  # type: np.float64
        qubit_state = ""
        for e in x_array:
            sign = '0' if e >= 0 else '1'

            e = abs(e)
            integer_part = int(e)

            decimal_part = e - integer_part
            decimal = []
            for d in range(self.decimal_length):
                decimal.append('1' if decimal_part*2 >= 1 else '0')
                decimal_part = decimal_part*2 - (1 if decimal_part*2 >= 1 else 0)

            qubit_state += sign + "{0:b}".format(integer_part).zfill(self.integer_length)[0:self.integer_length] + "".join(decimal)
        index_for = int(qubit_state, 2)
        feature_x[index_for, 0] = 1.0
        return feature_x



