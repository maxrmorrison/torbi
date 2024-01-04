from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from pybind11.setup_helpers import Pybind11Extension
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import numpy as np

import os
os.environ["CXX"] = "g++-11"
os.environ["CC"] = "gcc-11"

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

modules = [
    Extension(
        'ops',
        language='c',
        sources=['torbi/ops.pyx'],
        extra_compile_args=['-Ofast']
    ),
    Pybind11Extension(
        "fastops",
        ["torbi/ops.cpp"],
        extra_compile_args=[
            '-Ofast',
            '-fopenmp'
        ],
    ),
    CUDAExtension(
        'cudaops',
        [
            'torbi/ops_cuda.cpp',
            'torbi/ops_cuda_kernel.cu'
        ]
    )
]

setup(
    name='torbi',
    description='Cython-optimized Viterbi decoding and fast approximations',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torbi',
    # ext_modules=cythonize(
    #     ['torbi/ops.pyx'],
    #     compiler_directives={'language_level': '3'},
    # ),
    # ext_modules=cythonize(
    #     modules,
    #     compiler_directives={'language_level': '3'}
    # ),
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[np.get_include(), 'torbi'],
    setup_requires=['numpy', 'cython', 'torch'],
    install_requires=['numpy', 'torch', 'torchutil'],
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['cython', 'decode', 'sequence', 'torch', 'Viterbi'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
