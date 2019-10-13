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

from scipy import sparse

from . import EncodingMap


class IdentityEncodingMap(EncodingMap):
    def map(self, input_vector):
        # type: (IdentityEncodingMap, List[complex]) -> sparse.dok_matrix
        result = sparse.dok_matrix((len(input_vector), 1))
        for i, v in enumerate(input_vector):
            result[i, 0] = v
        return result