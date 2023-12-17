import math
from typing import Optional

import numpy as np
import torch
import torchutil

from torbi.ops import cforward


###############################################################################
# Viterbi decoding
###############################################################################


CYTHON = True


###############################################################################
# Viterbi decoding
###############################################################################


def decode(
    observation: torch.Tensor,
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None,
    max_chunk_size: Optional[int] = None,
    log_probs: bool = False
) -> torch.Tensor:
    """Perform Viterbi decoding on a sequence of distributions

    Arguments
        observation
            Sequence of log-categorical distributions
            shape=(frames, states)
        transition
            Log-categorical transition matrix
            shape=(states, states)
        initial
            Log-categorical initial distribution over states
            shape=(states,)
        max_chunk_size
            Size of each decoding chunk; O(nlogn) -> O(nlogk)
        log_probs
            Whether inputs are in (natural) log space

    Returns
        indices
            The decoded bin indices
            shape=(frames,)
    """
    torchutil.time.reset()

    with torchutil.time.context('setup'):
        frames, states = observation.shape

        # Cache device
        device = observation.device

        # Default to uniform initial probabilities
        if initial is None:
            initial = np.full(
                (frames,),
                math.log(1. / states),
                dtype=np.float32)

        # Ensure initial probabilities are in log space
        else:
            if not log_probs:
                initial = torch.log(initial)
            initial = initial.cpu().numpy().astype(np.float32)

        # Default to uniform transition probabilities
        if transition is None:
            transition = np.full(
                (frames, frames),
                math.log(1. / states),
                dtype=np.float32)

        # Ensure transition probabilities are in log space
        else:
            if not log_probs:
                transition = torch.log(transition)
            transition = transition.cpu().numpy().astype(np.float32)

        # Ensure observation probabilities are in log space
        if not log_probs:
            observation = torch.log(observation)
        observation = observation.cpu().numpy().astype(np.float32)

    # Chunked Viterbi decoding
    if max_chunk_size and max_chunk_size < len(observation):

        with torchutil.time.context('setup'):

            # Initialize intermediate arrays
            shape = (max_chunk_size, observation.shape[1])
            posterior = np.zeros(shape, dtype=np.float32)
            memory = np.zeros(shape, dtype=np.int32)

        indices = []
        for i in range(0, len(observation), max_chunk_size):
            size = min(len(observation) - i, max_chunk_size)

            with torchutil.time.context('forward'):

                # Forward pass of first chunk
                args = (
                    observation[i:i + max_chunk_size],
                    transition,
                    initial,
                    posterior,
                    memory)
                if CYTHON:
                    cforward(*args, size, states)
                else:
                    forward(*args)

            with torchutil.time.context('backward'):

                # Backward pass
                indices.append(backward(posterior, memory, size))

            with torchutil.time.context('next'):

                # Update initial to the posterior of the last chunk
                e_x = np.exp(posterior[-1] - np.max(posterior[-1]))
                initial = np.log(e_x / e_x.sum())

        print(torchutil.time.results())

        # Concatenate chunks
        indices = np.concatenate(indices)

    # No chunking
    else:

        with torchutil.time.context('setup'):

            # Initialize
            posterior = np.zeros_like(observation)
            memory = np.zeros(observation.shape, dtype=np.int32)

        with torchutil.time.context('forward'):

            # Forward pass
            args = (observation, transition, initial, posterior, memory)
            if CYTHON:
                cforward(*args, frames, states)
                # print(f'observation: {observation}')
                # print(f'transition: {transition}')
                # print(f'initial: {initial}')
                # print(f'posterior: {posterior}')
                # print(f'memory: {memory}')
            else:
                forward(*args)

        with torchutil.time.context('backward'):

            # Backward pass
            indices = backward(posterior, memory)

        print(torchutil.time.results())

    return torch.tensor(indices, dtype=torch.int, device=device)


###############################################################################
# Individual steps
###############################################################################


def backward(posterior, memory, size=None):
    """Get optimal pass from results of forward pass"""
    if size is None:
        size = len(posterior)

    # Initialize
    indices = np.full((size,), np.argmax(posterior[-1]), dtype=np.int32)

    # Backward
    for t in range(size - 2, -1, -1):
        indices[t] = memory[t + 1, indices[t + 1]]

    return indices


def forward(observation, transition, initial, posterior, memory):
    """Viterbi decoding forward pass"""
    # Add prior to first frame
    posterior[0] = observation[0] + initial

    # Forward pass
    # print(f'transition:\n {transition}')
    for t in range(1, observation.shape[0]):
        # print(f'posterior-{t - 1}:\n {posterior[t - 1]}')
        probability = posterior[t - 1] + transition
        # print(f'probability-{t}:\n {probability}')

        # Update best so far
        for j in range(observation.shape[1]):
            memory[t, j] = np.argmax(probability[j])
            posterior[t, j] = observation[t, j] + probability[j, memory[t, j]]

        # print(f'memory-{t}:\n {memory}')
    # print(f'posterior-{t}:\n {posterior}')

    return posterior, memory
