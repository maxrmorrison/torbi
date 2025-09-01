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
so_files = list(Path(__file__).parent.glob("_C*.so"))
assert (
    len(so_files) == 1
), f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

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
