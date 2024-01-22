import torch
import torbi

def chunk(sequence, min_chunk_size=64, entropy_threshold=torbi.ENTROPY_THRESHOLD):
    """Chunk sequence based on points of low entropy

    Arguments
        observation
            Time-varying categorical distribution
            shape=(frames, states)
        min_chunk_size
            Minimum chunk size in frames
        entropy_threshold
            Threshold for entropy to allow splitting


    Returns
        sub_sequences
            List of chunked sequence data
    """
    split_points = split(
        sequence=sequence,
        min_chunk_size=min_chunk_size,
        entropy_threshold=entropy_threshold
    )

    chunks = []

    start = 0
    for split_point in split_points:
        chunks.append(sequence[start:split_point])
        start = split_point
    chunks.append(sequence[start:]) # Grab last chunk

    return chunks

def split(sequence, min_chunk_size=256, entropy_threshold=torbi.ENTROPY_THRESHOLD):
    """Find split points of minimum entropy"""
    sequence = sequence.T
    length = sequence.shape[-1]
    candidates = entropy(sequence) < entropy_threshold
    split_points = []
    i = min_chunk_size
    while i < length:
        if candidates[i] and candidates[i - 1]:
            split_points.append(i)
            i += min_chunk_size
        else:
            i += 1
    return split_points

def entropy(sequence):
    """Compute the framewise categorical entropy"""
    return -(torch.exp(sequence) * sequence).sum(dim=0) / torch.log(torch.tensor(sequence.shape[0]))
