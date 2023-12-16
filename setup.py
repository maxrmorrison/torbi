from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import numpy as np


with open('README.md', encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='torbi',
    description='Cython-optimized Viterbi decoding and fast approximations',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torbi',
    ext_modules=cythonize(
        ['torbi/ops.pyx'],
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include(), 'torbi'],
    setup_requires=['numpy', 'cython'],
    install_requires=['numpy', 'torch', 'torchutil'],
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['cython', 'decode', 'sequence', 'torch', 'Viterbi'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
