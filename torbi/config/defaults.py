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
# Evaluation
###############################################################################


# Names of all datasets
DATASETS = ['daps']

# Number of randomly-selected samples to evaluate
EVALUATION_SAMPLES = 100

# Thresholds (in number of 5 cent bins) for raw pitch accuracy evaluation
PITCH_ERROR_THRESHOLDS = [0, 1, 2]

# File for caching transition matrix for pitch decoding evaluation
PITCH_TRANSITION_MATRIX = ASSETS_DIR / 'stats' / 'transition.pt'

# Seed for all random number generators
RANDOM_SEED = 1234


###############################################################################
# settings
###############################################################################

NUM_WORKERS = 0

BATCH_SIZE = 100