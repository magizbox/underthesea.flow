#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup, find_packages

# Use the VERSION file to get version
version_file = os.path.join(os.path.dirname(__file__), 'languageflow', 'VERSION')
with open(version_file) as fh:
    version = fh.read().strip()

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requirements = [
    'Click>=6.0',
    'requests',
    'tabulate',
    'clint',
    'tqdm',
    'scikit-learn==0.20.3',
    'joblib==0.13.2',
    'Cython',
    'fasttext==0.8.22'
]
dependency_links = [
    "git+ssh://git@github.com/facebookresearch/fastText.git@v0.2.0#egg=fasttext-v0.2.0"
]
setup_requirements = []

test_requirements = []

setup(
    name='languageflow',
    version=version,
    description="Data loaders and abstractions for text and NLP",
    long_description=readme + '\n\n' + history,
    author="Vu Anh",
    author_email='anhv.ict91@gmail.com',
    url='https://github.com/undertheseanlp/languageflow',
    packages=find_packages(include=['languageflow']),
    include_package_data=True,
    install_requires=install_requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='languageflow',
    entry_points={
        'console_scripts': [
            'languageflow=languageflow.cli:main'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    dependency_links=dependency_links
)
