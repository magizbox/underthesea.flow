#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requirements = [
    'flair==0.4.1',
    'Click>=6.0',
    'requests',
    'tabulate',
    'clint',
    'tqdm'
]

setup_requirements = []

test_requirements = []

setup(
    name='languageflow',
    version='1.1.10-rc',
    description="Useful stuffs for NLP experiments",
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
)
