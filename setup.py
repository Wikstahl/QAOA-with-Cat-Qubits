#!/usr/bin/env python
"""
Setup file
"""
from setuptools import setup, find_packages

setup(name='ptm',
      version='0.0.1',
      description='Pauli transfer matrix',
      author='Pontus Vikstål',
      packages = find_packages(include=['ptm', 'ptm.*'])
     )
