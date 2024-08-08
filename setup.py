import os
from setuptools import find_packages, setup
import numpy as np
from torch.utils.cpp_extension import BuildExtension, CppExtension

use_cuda = os.environ.get('TORBI_USE_CUDA', 1)

if use_cuda == 1:
    from torch.utils.cpp_extension import CUDAExtension
    os.environ['CXX'] = 'g++-11'
    os.environ['CC'] = 'gcc-11'
    modules = [
        CUDAExtension(
            'viterbi',
            [
                'torbi/viterbi.cpp',
                'torbi/viterbi_kernel.cu'
            ],
            # extra_compile_args={'cxx': [], 'nvcc': ['-keep', '-G', '-O3', '--source-in-ptx']}
            extra_compile_args={'cxx': ['-fopenmp', '-O3'], 'nvcc': ['-O3']}
        )
    ]
else:
    modules = [
        CppExtension(
            'viterbi',
            [
                'torbi/viterbi_cpu.cpp'
            ],
            extra_compile_args={'cxx': ['-fopenmp', '-O3']}
        )
    ]


setup(
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[np.get_include(), 'torbi'],
    packages=find_packages(),
)
