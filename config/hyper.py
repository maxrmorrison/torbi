from pathlib import Path
from itertools import product

import torbi

MODULE = 'torbi'

if hasattr(torbi, "defaults"):

    progress_file = Path(__file__).parent / 'hyper.progress'

    # if not torbi.CONFIG_ONCE:
    if not progress_file.exists():
        progress = 0
    else:
        with open(progress_file) as f:
            progress = int(f.read())

    with open(progress_file, "w+") as f:
        f.write(str(progress+1))
        
    # torbi.CONFIG_ONCE = True
    batch_size = [16, 32, 64, 128, 256, 512]

    min_chunk_size = [64, 128, 256, 512, 1024]

    entropy_threshold = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # entropy_threshold = [0.35, 0.375, 0.4, 0.425, 0.45]

    combinations = list(product(batch_size, min_chunk_size, entropy_threshold))

    total_count = len(combinations)

    if progress >= total_count:
        raise IndexError("Done!")

    batch_size, min_chunk_size, entropy_threshold = combinations[progress]

    print(f"progress: {progress}/{total_count}")
    # print(batch_size, min_chunk_size, entropy_threshold)

    CONFIG = f"hyper-{batch_size}-{min_chunk_size}-{entropy_threshold}".replace('.', '_')
    # print(CONFIG)

    BATCH_SIZE = batch_size
    MIN_CHUNK_SIZE = min_chunk_size
    ENTROPY_THRESHOLD = entropy_threshold

    COMPARE_WITH_REFERENCE = False