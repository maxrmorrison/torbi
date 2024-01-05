import yapecs

import torbi


###############################################################################
# Download datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = yapecs.ArgumentParser(description='Download datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=torbi.DATASETS,
        default=torbi.DATASETS,
        help='The datasets to download')
    return parser.parse_args()


torbi.data.download.datasets(**vars(parse_args()))
