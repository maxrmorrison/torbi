###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('torbi', defaults)

# Import configuration parameters
from .config.defaults import *
import torbi
del torbi.defaults # remove unnecessary module
from .config.static import *


###############################################################################
# Module imports
###############################################################################

import torch
# load the library .so
# taken from https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
from pathlib import Path
import platform
if platform.system() == "Windows":
    lib_files = list(Path(__file__).parent.glob("_C*.pyd"))
else:
    lib_files = list(Path(__file__).parent.glob("_C*.so"))

if len(lib_files) == 0: # no lib files found
    raise FileNotFoundError('No library files found (.so or .pyd). Your torbi install is somehow corrupt')
elif len(lib_files) == 1: # one lib file found
    torch.ops.load_library(lib_files[0])
else: # multiple lib files, choose the one that corresponds to torch/cuda version
    # get torch, cuda versions
    torch_version = torch.__version__.split('+')[0]
    torch_version = ''.join(torch_version.split('.')[:-1])
    cuda_version = torch.version.cuda
    if cuda_version is None:
        cuda_version = 'cpu'
    else:
        cuda_version = 'cu' + cuda_version.replace('.', '')

    # create ptXXcuXXX version string
    binary_version_string = f'pt{torch_version}{cuda_version}'

    # Find the matching library file
    lib_file_pattern = f'_C.{binary_version_string}*'
    lib_candidates = list(Path(__file__).parent.glob(lib_file_pattern))
    if len(lib_candidates) == 0:
        raise FileNotFoundError(f"Could not find any lib files matching version {binary_version_string}")
    elif len(lib_candidates) > 1:
        raise RuntimeError(f"Found {len(lib_candidates)} lib files matching version {binary_version_string}, expected only 1")
    else:
        torch.ops.load_library(lib_candidates[0])


if torch.backends.mps.is_available():
    import warnings
    try:
        import ninja
        from . import mps
    except ImportError:
        warnings.warn("ninja is not installed, so the mps backend cannot be loaded.")
    except RuntimeError as e:
        warnings.warn("Could not compile mps backend:", e)

from .viterbi import decode
from .core import *
from .chunk import chunk
from . import data
from . import evaluate
from . import partition
from . import reference
