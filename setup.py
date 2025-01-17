#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


with open(os.path.join('networkxcywebanalyzer', '__init__.py')) as ver_file:
    for line in ver_file:
        if line.startswith('__version__'):
            version=re.sub("'", "", line[line.index("'"):])

requirements = [
    'ndex2>=3.9.0',
    'networkx>=3.0,<3.5'
]

test_requirements = [
    'requests-mock'
    # TODO: put package test requirements here
]

setup(
    name='networkxcywebanalyzer',
    version=version,
    description="Maps genes to terms",
    long_description=readme + '\n\n' + history,
    author="Chris Churas",
    author_email='cchuras@ucsd.edu',
    url='https://github.com/idekerlab/networkx_cyweb_analyzer',
    packages=[
        'networkxcywebanalyzer',
    ],
    package_dir={'networkxcywebanalyzer':
                 'networkxcywebanalyzer'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    keywords='networkxcywebanalyzer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    scripts=['networkxcywebanalyzer/networkxcywebanalyzercmd.py'],
    test_suite='tests',
    tests_require=test_requirements
)
