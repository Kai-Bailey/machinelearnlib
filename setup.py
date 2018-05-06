from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(name='machinelearnlib',
      version='0.2',
      description='Implementation of common machine learning algorithms',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/Kai-Bailey/machinelearnlib',
      author='Kai Bailey',
      author_email='kbailey1@ualberta.ca',
      license='MIT',
      keywords='machine learning data science',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=['numpy', 'matplotlib'],
      entry_points={'console_scripts':['ml=machinelearnlib.testing:testing']})