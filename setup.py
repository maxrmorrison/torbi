import os
from setuptools import find_packages, setup
import numpy as np
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)
import platform
from pathlib import Path
import glob
import warnings

library_name = "torbi"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

windows = platform.system() == "Windows"
macos = platform.system() == "Darwin"

if windows:
    cxx_args = ['/O2', '/openmp']
elif macos:
    cxx_args = ['-O3', '-Xclang', '-fopenmp', '-L/opt/homebrew/opt/libomp/lib', '-I/opt/homebrew/opt/libomp/include', '-lomp']
else:
    cxx_args = ['-fopenmp', '-O3']

# if use_cuda:
#     from torch.utils.cpp_extension import CUDAExtension
#     os.environ['CXX'] = 'g++-11'
#     os.environ['CC'] = 'gcc-11'
#     modules = [
#         CUDAExtension(
#             'viterbi',
#             [
#                 'torbi/viterbi.cpp',
#                 'torbi/viterbi_kernel.cu'
#             ],
#             # extra_compile_args={'cxx': [], 'nvcc': ['-keep', '-G', '-O3', '--source-in-ptx']}
#             extra_compile_args={'cxx': cxx_args, 'nvcc': ['-O3', '-allow-unsupported-compiler']}
#         )
#     ]
# else:
#     modules = [
#         CppExtension(
#             'viterbi',
#             [
#                 'torbi/viterbi_cpu.cpp'
#             ],
#             extra_compile_args={'cxx': cxx_args}
#         )
#     ]


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if use_cuda:
        warnings.warn(f"Building torbi with CUDA using CUDA_HOME={CUDA_HOME}")

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-fopenmp",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules

setup(
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    include_dirs=[np.get_include(), 'torbi'],
    packages=find_packages(),
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
