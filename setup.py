import os
from setuptools import find_packages, setup
import numpy as np
from torch.utils.cpp_extension import BuildExtension, CppExtension
import platform
from pathlib import Path

windows = platform.system() == "Windows"
macos = platform.system() == "Darwin"

if macos:
    use_cuda = False # No cuda drivers anyway

    #also check for homebrew libomp
    libomp_dir = Path('/opt/homebrew/opt/libomp/')
    if not libomp_dir.exists():
        raise FileNotFoundError("On MacOS, you must install libomp through homebrew (brew install libomp)")
else:
    use_cuda = bool(int(os.environ.get('TORBI_USE_CUDA', 1)))

if windows:
    cxx_args = ['/O2', '/openmp']
elif macos:
    cxx_args = ['-O3', '-Xclang', '-fopenmp', '-L/opt/homebrew/opt/libomp/lib', '-I/opt/homebrew/opt/libomp/include', '-lomp']
else:
    cxx_args = ['-fopenmp', '-O3']

if use_cuda:
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
            extra_compile_args={'cxx': cxx_args, 'nvcc': ['-O3', '-allow-unsupported-compiler']}
        )
    ]
else:
    modules = [
        CppExtension(
            'viterbi',
            [
                'torbi/viterbi_cpu.cpp'
            ],
            extra_compile_args={'cxx': cxx_args}
        )
    ]

setup(
    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[np.get_include(), 'torbi'],
    packages=find_packages(),
)
