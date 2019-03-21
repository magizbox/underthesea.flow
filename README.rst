============
LanguageFlow
============

.. image:: https://img.shields.io/pypi/v/languageflow.svg
        :target: https://pypi.python.org/pypi/languageflow

.. image:: https://img.shields.io/pypi/pyversions/languageflow.svg
        :target: https://pypi.python.org/pypi/languageflow

.. image:: https://img.shields.io/badge/license-GNU%20General%20Public%20License%20v3-brightgreen.svg
        :target: https://pypi.python.org/pypi/languageflow

.. image:: https://img.shields.io/travis/undertheseanlp/languageflow.svg
        :target: https://travis-ci.org/undertheseanlp/languageflow

.. image:: https://readthedocs.org/projects/languageflow/badge/?version=latest
        :target: http://languageflow.readthedocs.io/en/latest/
        :alt: Documentation Status

Data loaders and abstractions for text and NLP

Requirements
------------

Install dependencies

.. code-block:: bash

    $ pip install future, tox, joblib
    $ pip install numpy scipy pandas scikit-learn==0.19.1
    $ pip install python-crfsuite==0.9.5
    $ pip install Cython
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ pip install xgboost


Installation
------------

.. code-block:: bash

    $ pip install languageflow

Components
------------

* Transformers: NumberRemover, CountVectorizer, TfidfVectorizer
* Models: SGDClassifier, XGBoostClassifier, KimCNNClassifier, FastTextClassifier, CRF

Data
------------

Download a dataset using **download** command

.. code-block:: bash

    $ languageflow download [DATA]


Datasets
~~~~~~~~

The datasets module currently contains:

* PlaintextCorpus: VNESES, VNTQ_SMALL, VNTQ_BIG
