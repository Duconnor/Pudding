from ctypes import pythonapi
import sys, os
from setuptools import setup

# Package metadata
NAME = 'pudding'
DESCRIPTION = 'Python binding of the Pudding library'
URL = 'https://github.com/Duconnor/Pudding'
EMAIL = 'duconnor@outlook.com'
AUTHOR = 'Yingxiao Du'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# Required packages
REQUIRED = [

]

# Optional packages
EXTRA = [

]

path_here = os.path.dirname(os.path.abspath(__file__))
# Try to read the content in the README as the long description
try:
    with open(os.path.join(path_here, 'README.md'), 'r') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    license='MIT',
    packages=[NAME],
    install_requires=REQUIRED,
)