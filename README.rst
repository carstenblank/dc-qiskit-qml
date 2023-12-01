Data Cybernetics qiskit-qml
############################

.. image:: https://img.shields.io/travis/com/carstenblank/dc-qiskit-qml/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.com/carstenblank/dc-qiskit-qml

.. image:: https://img.shields.io/codecov/c/github/carstenblank/dc-qiskit-qml/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/carstenblank/dc-qiskit-qml

.. image:: https://img.shields.io/codacy/grade/820b74d1739b4d31b6395bfd8469b3bb.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://www.codacy.com/app/carstenblank/dc-qiskit-qml?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=carstenblank/dc-qiskit-qml&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/readthedocs/dc-qiskit-qml.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://dc-qiskit-qml.readthedocs.io

.. image:: https://img.shields.io/pypi/v/dc-qiskit-qml.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/dc-qiskit-qml

.. image:: https://img.shields.io/pypi/pyversions/dc-qiskit-qml.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/dc-qiskit-qml

.. header-start-inclusion-marker-do-not-remove

`qiskit <https://qiskit.org/documentation/>`_ is an open-source compilation framework capable of targeting various
types of hardware and a high-performance quantum computer simulator with emulation capabilities and various
compiler plug-ins.

This library implements so far one quantum machine learning classifier which has been introduced by F.Petruccione,
M. Schuld and M. Fingerhuth (http://stacks.iop.org/0295-5075/119/i=6/a=60002). Athough this is the only classifier
implemented so far, this library is to be used as a repository for more classifiers using qiskit as a background
framework.


Features
========

* Distance & Majority based Hadamard-gate classifier

    * Generic real valued vector space input data (slow)

    * Binary valued vector space input data (faster)

    * Feature map pre-processing for non-linear classification

.. header-end-inclusion-marker-do-not-remove

.. installation-start-inclusion-marker-do-not-remove

Installation
============

This library requires Python version 3.5 and above, as well as qiskit.
Installation of this library, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install dc_qiskit_qml

To test that the algorithms are working correctly you can run

.. code-block:: bash

    $ make test

.. installation-end-inclusion-marker-do-not-remove

.. gettingstarted-start-inclusion-marker-do-not-remove

Getting started
===============

You can check out the classifier as follows

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    import qiskit

    from dc_qiskit_qml.feature_maps import NormedAmplitudeEncoding
    from dc_qiskit_qml.distance_based.hadamard import QmlHadamardNeighborClassifier
    from dc_qiskit_qml.distance_based.hadamard.state import QmlGenericStateCircuitBuilder
    from dc_qiskit_qml.distance_based.hadamard.state.sparsevector import MottonenStatePreparation

    X, y = load_iris(True)
    # Only the first two features and only get two labels
    # This is a toy example!
    X = np.asarray([x[0:2] for x, yy in zip(X, y) if yy != 2])
    y = np.asarray([yy for x, yy in zip(X, y) if yy != 2])

    preprocessing_pipeline = Pipeline([
        ('scaler',  StandardScaler()),
        ('l2norm', Normalizer(norm='l2', copy=True))
    ])
    X = preprocessing_pipeline.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Using the generic wave function (state vector) routine using the 'Möttönen'
    # state preparation algorithm
    initial_state_builder = QmlGenericStateCircuitBuilder(MottonenStatePreparation())

    # The normed amplitude encoding ensures that the data is normalized
    # This is a somewhat unnecessary step as above we do that already
    feature_map = NormedAmplitudeEncoding()

    execution_backend: BaseBackend = qiskit.Aer.get_backend('qasm_simulator')
    qml = QmlHadamardNeighborClassifier(backend=execution_backend,
                                        shots=8192,
                                        classifier_circuit_factory=initial_state_builder,
                                        feature_map=feature_map)

    qml.fit(X_train, y_train)
    prediction = qml.predict(X_test)

    "Test Accuracy: {}".format(
        sum([1 if p == t else 0 for p, t in zip(prediction, y_test)])/len(prediction)
    )

    prediction_train = qml.predict(X_train)
    "Train Accuracy: {}".format(
        sum([1 if p == t else 0 for p, t in zip(prediction_train, y_train)])/len(prediction_train)
    )

The details are a bit more involved as to how this works and the classifier can be configured with a circuit factory
or a feature map.

.. gettingstarted-end-inclusion-marker-do-not-remove

Please refer to the `documentation of the dc qiskit qml library <https://dc-qiskit-qml.readthedocs.io/>`_ .

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane.

Authors
=======

Carsten Blank

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/carstenblank/dc-qiskit-qml
- **Issue Tracker:** https://github.com/carstenblank/dc-qiskit-qml/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The data cybernetics qiskit algorithms plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
