from pathlib import Path
from itertools import product
import torch

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
    # batch_size = [16, 32, 64, 128, 256, 512, 1024]
    # batch_size = range(128, 256+1, 8)
    # batch_size = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256] + list(range(256, 2048+1, 32))
    batch_size = list(range(1024, 2048, 128))

    min_chunk_size = [torch.inf]
    # min_chunk_size = [64, 128, 256, 512, 1024]

    entropy_threshold = [0.0]

    # entropy_threshold = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # entropy_threshold = [0.35, 0.375, 0.4, 0.425, 0.45]

    combinations = list(product(batch_size, min_chunk_size, entropy_threshold))

    total_count = len(combinations)

    if progress >= total_count:
        raise IndexError("Done!")

    batch_size, min_chunk_size, entropy_threshold = combinations[progress]

    print(f"progress: {progress}/{total_count}")
    # print(batch_size, min_chunk_size, entropy_threshold)

    # CONFIG = f"cpu-{batch_size}-{min_chunk_size}-{entropy_threshold}".replace('.', '_')
    CONFIG = f"cuda-{batch_size}".replace('.', '_')
    # print(CONFIG)

    BATCH_SIZE = batch_size
    MIN_CHUNK_SIZE = min_chunk_size
    ENTROPY_THRESHOLD = entropy_threshold

    # COMPARE_WITH_REFERENCE = False