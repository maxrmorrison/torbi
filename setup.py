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

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    if use_cuda:
        warnings.warn(f"Building torbi with CUDA using CUDA_HOME={CUDA_HOME}")

    windows = platform.system() == "Windows"
    macos = platform.system() == "Darwin"

    if windows:
        cxx_args = [
            '/O2',
            '/openmp'
            '/D Py_LIMITED_API=0x03090000' # min CPython version 3.9
        ]
        if debug_mode:
            raise ValueError('debug_mode not currently supported on windows')
    elif macos:
        cxx_args = [
            '-Xclang',
            '-fopenmp',
            # '-L/opt/homebrew/opt/libomp/lib',
            # '-I/opt/homebrew/opt/libomp/include',
            # '-lomp',
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ]
    else: # linux
        cxx_args = [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-fopenmp",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ]

    extra_link_args = []
    extra_compile_args = {
        "cxx": cxx_args,
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
