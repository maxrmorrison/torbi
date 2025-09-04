from setup import get_extensions
from torch.utils.cpp_extension import BuildExtension
from setuptools import dist
from pathlib import Path
from shutil import copy as cp
from shutil import rmtree as rm
import torch

# build c++/CUDA extension
d = dist.Distribution({
    "ext_modules": get_extensions(),
    "cmdclass": {"build_ext": BuildExtension}
})
cmd = d.get_command_obj("build_ext")
cmd.ensure_finalized()
cmd.run()

# get torch, cuda versions
torch_version = torch.__version__.split('+')[0]
torch_version = ''.join(torch_version.split('.')[:-1])
cuda_version = torch.version.cuda
if cuda_version is None:
    cuda_version = 'cpu'
cuda_version = cuda_version.replace('.', '')

# create ptXXcuXXX version string
binary_version_string = f'pt{torch_version}cu{cuda_version}'
print(binary_version_string)

# locate binary file
binary_file = Path(cmd.get_outputs()[0])
assert binary_file.exists()

binary_name_parts = binary_file.name.split('.')
binary_name_parts.insert(1, binary_version_string)
new_binary_name = '.'.join(binary_name_parts)

cp(binary_file, Path('build') / new_binary_name)
# rm(binary_file.parent)