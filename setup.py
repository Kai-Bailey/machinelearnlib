from setuptools import setup, find_packages
from os import path

setup(name='machinelearnlib',
      version='0.2.7',
      description='Implementation of common machine learning algorithms',
      url='https://github.com/Kai-Bailey/machinelearnlib',
      author='Kai Bailey',
      author_email='kbailey1@ualberta.ca',
      license='MIT',
      keywords='machine learning data science',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=['numpy', 'matplotlib'],)