#!/usr/bin/env python
"""
Setup file
"""
from setuptools import setup, find_packages

setup(name='cvqaoa',
      version='0.0.1',
      description='Pauli transfer matrix',
      author='Pontus Vikstål',
      author_email='pontus.wikstahl@gmail.com',
      packages = find_packages(include=['cvqaoa', 'cvqaoa.*'])
     )
