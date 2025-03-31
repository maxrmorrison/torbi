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
from . import _C
from .viterbi import decode
from .core import *
from .chunk import chunk
from . import data
from . import evaluate
from . import partition
from . import reference
