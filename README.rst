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

⚠️ **WARNING**
-------------

⚠️ **WARNING** This project is **archived** and **no longer maintained** **WARNING** ⚠️

⏩ From **underthesea v1.2.0**, languageflow is merged with underthesea. So enjoy a better ``underthesea`` *without languageflow*.

⚰️ **RIP Languageflow**

Old Readme
----------

Data loaders and abstractions for text and NLP

Requirements
------------

Install dependencies

.. code-block:: bash

    $ pip install future, tox
    $ pip install python-crfsuite==0.9.5
    $ pip install Cython
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ pip install xgboost==0.82


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

    $ languageflow download DATASET

List all dataset

.. code-block:: bash

    $ languageflow list


Datasets
~~~~~~~~

The datasets module currently contains:

* Tagged: VLSP2018-NER, VTB-CHUNK*, VLSP2016-NER*, VLSP2013-POS*, VLSP2013-WTK*
* Categorized: AIVIVN2019_SA*, VLSP2018_SA*, UTS2017_BANK, VLSP2016_SA*, VNTC
* Plaintext: VNESES, VNTQ_SMALL, VNTQ_BIG

Caution (*): With closed license dataset, you must provide URL to download

Example
~~~~~~~~

Download ``UTS2017_BANK`` dataset

.. code-block:: bash

    $ languageflow download UTS2017_BANK

Use ``UTS2017_BANK`` dataset

.. code-block:: python

    >>> from languageflow.data_fetcher import DataFetcher, NLPData
    >>> corpus = DataFetcher.load_corpus(NLPData.UTS2017_BANK_SA)
    >>> print(corpus)
    CategorizedCorpus: 1780 train + 197 dev + 494 test sentences
