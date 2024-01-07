import json
import random

import torbi


###############################################################################
# Partition
###############################################################################


def datasets(datasets):
    """Partition datasets and save to disk"""
    for dataset in datasets:
        random.seed(torbi.RANDOM_SEED)

        # Get stems
        directory = torbi.CACHE_DIR / dataset
        stems = [
            f'{file.parent.name}/{file.stem}'
            for file in directory.rglob('*.wav')]

        # Shuffle
        random.shuffle(stems)

        # Slice
        stems = stems[:torbi.EVALUATION_SAMPLES]

        # Save to disk
        file = torbi.PARTITION_DIR / f'{dataset}.json'
        file.parent.mkdir(exist_ok=True, parents=True)
        with open(file, 'w') as file:
            json.dump(stems, file, indent=4)
