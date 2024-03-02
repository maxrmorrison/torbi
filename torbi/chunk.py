from typing import List

import torch
import torbi


###############################################################################
# Chunked Viterbi decoding; quickly and accurately batch process long sequences
###############################################################################


def chunk(
    observation: torch.Tensor,
    min_chunk_size: int = torbi.MIN_CHUNK_SIZE,
    entropy_threshold: float = torbi.ENTROPY_THRESHOLD
) -> List:
    """Chunk observations based on points of low entropy

    Arguments
        observation
            Time-varying categorical distribution
            shape=(frames, states)
        min_chunk_size
            Minimum chunk size in frames
        entropy_threshold
            Threshold for entropy to allow splitting

    Returns
        chunks
            List of chunked sequence data
    """
    start = 0
    chunks = []

    # Get split points
    for split_point in split(
        observation,
        min_chunk_size=min_chunk_size,
        entropy_threshold=entropy_threshold
    ):

        # Chunk observations
        chunks.append(observation[start:split_point])
        start = split_point

     # Get last chunk
    chunks.append(observation[start:])

    return chunks


###############################################################################
# Utilities
###############################################################################


def split(
    observation,
    min_chunk_size=torbi.MIN_CHUNK_SIZE,
    entropy_threshold=torbi.ENTROPY_THRESHOLD
) -> List[int]:
    """Find split points of minimum entropy"""
    observation = observation.T

    # Find low-entropy time frames
    candidates = entropy(observation) < entropy_threshold

    # Find adjacent low-entropy time frames
    split_points = []
    i = min_chunk_size
    while i < observation.shape[-1]:
        if candidates[i] and candidates[i - 1]:
            split_points.append(i)
            i += min_chunk_size
        else:
            i += 1

    return split_points


def entropy(observation: torch.Tensor) -> torch.Tensor:
    """Compute framewise entropy of a sequence of categorical distributions"""
    return -(
        (torch.exp(observation) * observation).sum(dim=0) /
        torch.log(torch.tensor(observation.shape[0])))
