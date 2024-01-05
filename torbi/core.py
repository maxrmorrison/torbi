import math
from typing import Optional

import numpy as np
import torch
import torchutil

from ops import cforward
from .pytorch import tforward
#TODO fix this name
from fastops import cppforward
from cudaops import forward as cuda_forward


###############################################################################
# Viterbi decoding
###############################################################################


CYTHON = False
TORCH = False
PYBIND = False
CUDA = False

# Easier way to choose method
CUDA = True


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
                    # raise ValueError('removed')
                elif TORCH:
                    print('copying to GPU')
                    device = 'cuda:0' #TODO fix
                    observation = torch.tensor(observation, device=device)
                    transition = torch.tensor(transition, device=device)
                    initial = torch.tensor(initial, device=device)
                    posterior = torch.tensor(posterior, device=device)
                    memory = torch.tensor(memory, device=device)
                    probability = torch.tensor(probability, device=device)
                    args = (observation, transition, initial, posterior, memory, probability)
                    print('about to do tforward')
                    with torch.inference_mode():
                        tforward(*args, frames, states)
                    posterior = posterior.cpu().numpy()
                    memory = memory.cpu().numpy()
                elif PYBIND:
                    print('starting c++ decode')
                    cppforward(*args, frames, states)
                elif CUDA:
                    print('copying to GPU')
                    device = 'cuda:0' #TODO fix
                    observation = torch.tensor(observation, device=device)
                    transition = torch.tensor(transition, device=device)
                    initial = torch.tensor(initial, device=device)
                    posterior = torch.tensor(posterior, device=device)
                    memory = torch.tensor(memory, device=device)
                    probability = torch.tensor(probability, device=device)
                    args = (observation, transition, initial, posterior, memory, probability)
                    print('about to do cuda forward')
                    cuda_forward(*args, frames, states)
                    posterior = posterior.cpu().numpy()
                    memory = memory.cpu().numpy()
                else:
                    forward(*args)
                # print(f'observation:\n{observation}')
                # print(f'transition:\n{transition}')
                # print(f'initial:\n{initial}')
                print(f'posterior:\n{posterior}')
                print(f'memory:\n{memory}')

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
