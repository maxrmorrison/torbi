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
    log_probs: bool = False
) -> torch.Tensor:
    """Perform Viterbi decoding on a sequence of distributions

    Arguments
        observation
            Time-varying log-categorical distribution on the desired device
            shape=(frames, states)
        transition
            Log-categorical transition matrix
            shape=(states, states)
        initial
            Log-categorical initial distribution over states
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space

    Returns
        indices
            The decoded bin indices
            shape=(frames,)
    """
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

    with torchutil.time.context('setup'):

        # Initialize
        posterior = np.zeros_like(observation)
        memory = np.zeros(observation.shape, dtype=np.int32)
        probability = np.zeros((states, states), dtype=np.float32)

    with torchutil.time.context('forward'):

        # Forward pass
        args = (observation, transition, initial, posterior, memory, probability)
        if CYTHON:
            cforward(*args, frames, states)
        else:
            forward(*args)

    with torchutil.time.context('backward'):

        # Backward pass
        indices = backward(posterior, memory)

    return torch.tensor(indices, dtype=torch.int, device=device)


###############################################################################
# Utilities
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
