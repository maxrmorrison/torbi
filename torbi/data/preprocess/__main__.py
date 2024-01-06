import yapecs

import torbi


###############################################################################
# Preprocess datasets
###############################################################################


def parse_args():
    parser = yapecs.ArgumentParser(description='Preprocess datasets')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=torbi.DATASETS,
        choices=torbi.DATASETS,
        help='The datasets to preprocess')
    parser.add_argument(
        '--gpu',
        type=int,
        help='The index of the gpu to use')
    return parser.parse_args()

if __name__ == '__main__':
    torbi.data.preprocess.datasets(**vars(parse_args()))
