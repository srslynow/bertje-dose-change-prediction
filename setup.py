#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages

exclude = []

setup(
    name='caracal-dosechange-prediction',
    version='1.0',
    description='caracal dosechange prediction',
    author='Koen Vijverberg',
    author_email='koen.vijverberg@caracal.nl',
    packages=find_namespace_packages(include=['caracal.*'], exclude=exclude),
    include_package_data=True,
)
