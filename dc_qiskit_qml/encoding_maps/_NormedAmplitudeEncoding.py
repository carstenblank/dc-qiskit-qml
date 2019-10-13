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


import logging

import numpy as np
from scipy import sparse

from . import EncodingMap

log = logging.getLogger('VectorAmplitudeEncoding')


class NormedAmplitudeEncoding(EncodingMap):
    def map(self, x):
        # type: (NormedAmplitudeEncoding, any) -> sparse.dok_matrix
        x_array = np.asarray(x)
        x_norm = np.linalg.norm(x_array)
        x_array = x_array / x_norm
        log.info("Normed Input Vector: %s" % x_array)
        feature_x = sparse.dok_matrix((x_array.size, 1), dtype=complex)  # type: sparse.dok_matrix
        for i, e in enumerate(x_array):
            feature_x[i, 0] = e/x_norm
        return feature_x
