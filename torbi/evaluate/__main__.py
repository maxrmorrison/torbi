import yapecs
from pathlib import Path

import torbi


###############################################################################
# Evaluate
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Perform evaluation')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=torbi.DATASETS,
        help='The datasets to evaluate')
    parser.add_argument(
        '--gpu',
        help='The index of the gpu to use for evaluation, or "mps"')
    parser.add_argument(
        '--num_threads',
        type=int,
        default = 1,
        help='The number of threads to use for parellelized CPU decoding'
    )
    return parser.parse_args()


torbi.evaluate.datasets(**vars(parse_args()))
