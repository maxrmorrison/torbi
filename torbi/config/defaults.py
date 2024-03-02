from pathlib import Path


###############################################################################
# Metadata
###############################################################################


# Configuration name
CONFIG = 'torbi'


###############################################################################
# Directories
###############################################################################


# Root location for saving outputs
ROOT_DIR = Path(__file__).parent.parent.parent

# Location to save assets to be bundled with pip release
ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# Location of preprocessed features
CACHE_DIR = ROOT_DIR / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Location to save evaluation artifacts
EVAL_DIR = ROOT_DIR / 'eval'


###############################################################################
# Decoding
###############################################################################


# When set to a positive integer, enables chunking for long sequences by
# splitting sequences at low-entropy frames
MIN_CHUNK_SIZE = None

# Threshold below which to split the sequence when performing chunked decoding
ENTROPY_THRESHOLD = 0.5


###############################################################################
# Evaluation
###############################################################################


# Otherwise compare against self with no chunking
COMPARE_WITH_REFERENCE = True

# Names of all datasets
DATASETS = ['daps', 'vctk']

# Number of randomly-selected samples to evaluate
EVALUATION_SAMPLES = 8192

# Thresholds (in number of 5 cent bins) for raw pitch accuracy evaluation
PITCH_ERROR_THRESHOLDS = [0, 1, 2]

# File for caching transition matrix for pitch decoding evaluation
PITCH_TRANSITION_MATRIX = ASSETS_DIR / 'stats' / 'transition.pt'

# Audio sampling rate
SAMPLE_RATE = 16000

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# Compute
###############################################################################


# Batch size
BATCH_SIZE = 512

# Number of parallel CPU workers
NUM_WORKERS = 0


###############################################################################
# Metadata
###############################################################################


# Allows config files to detect if this module is being configured
CONFIGURING = None
