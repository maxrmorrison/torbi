import yapecs

import torbi


###############################################################################
# Partition datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Partition datasets')
    parser.add_argument(
        '--datasets',
        default=torbi.DATASETS,
        nargs='+',
        help='The datasets to partition')
    return parser.parse_args()


torbi.partition.datasets(**vars(parse_args()))
