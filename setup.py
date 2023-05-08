#!/usr/bin/env python
"""
Setup file
"""
from setuptools import setup, find_packages

setup(name='qaoa_with_cat_qubits',
      version='0.0.1',
      description='QAOA with Cat Qubits',
      author='Pontus Vikstål',
      author_email='pontus.wikstahl@gmail.com',
      packages = find_packages(include=['qaoa_with_cat_qubits', 'qaoa_with_cat_qubits.*'])
     )
