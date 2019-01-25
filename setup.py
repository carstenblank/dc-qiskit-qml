#!/usr/bin/env python
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
from setuptools import setup

with open("dc_qiskit_qml/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open("requirements.txt") as f:
    requirements = f.readlines()

info = {
    'name': 'dc_qiskit_qml',
    'version': version,
    'maintainer': 'Carsten Blank',
    'maintainer_email': 'blank@data-cybernetics.com',
    'url': 'http://data-cybernetics.com',
    'license': 'Apache License 2.0',
    'packages': [
        'dc_qiskit_qml',
        'dc_qiskit_qml.distance_based',
        'dc_qiskit_qml.distance_based.hadamard',
        'dc_qiskit_qml.distance_based.hadamard.state',
        'dc_qiskit_qml.distance_based.hadamard.state.cnot',
        'dc_qiskit_qml.distance_based.hadamard.state.sparsevector',
        'dc_qiskit_qml.feature_maps'
    ],
    'description': 'Machine learning (quantum-)algorithms with qiskit as basis',
    'long_description': open('README.rst').read(),
    'provides': ["dc_qiskit_algorithms"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
