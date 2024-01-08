import os
from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

import numpy as np
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


os.environ['CXX'] = 'g++-11'
os.environ['CC'] = 'gcc-11'


with open('README.md', encoding='utf-8') as file:
    long_description = file.read()


modules = [
    Pybind11Extension(
        'fastops',
        ['torbi/ops.cpp'],
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
        ],
        # extra_compile_args={'cxx': [], 'nvcc': ['-keep', '-G', '-O3', '--source-in-ptx']}
    )
]


setup(
    name='torbi',
    description='Optimized Viterbi decoding and fast approximations',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torbi',
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[np.get_include(), 'torbi'],
    setup_requires=['numpy', 'pybind11', 'torch'],
    install_requires=['numpy', 'torch', 'torchutil', 'yapecs'],
    extras_require={'evaluate': ['librosa', 'torchaudio']},
    packages=find_packages(),
    package_data={'torbi': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['decode', 'sequence', 'torch', 'Viterbi'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
