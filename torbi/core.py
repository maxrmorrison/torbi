from typing import Optional

import torch


###############################################################################
# Viterbi decoding
###############################################################################


def decode(
    observation: torch.Tensor,
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Perform Viterbi decoding on a sequence of categorical distributions
    
    Args:
        observation: shape=(frames, states)
            Sequence of log-categorical distributions
        transition: shape=(states, states)
            Log-categorical transition matrix
        initial: shape=(states,)
            Log-categorical initial distribution over states

    Returns:
        indices: shape=(frames,)
            The decoded bin indices
    """
    frames, states = observation.shape
    device = observation.device

    # Default to uniform initial and transition probabilities
    if initial is None:
        initial = torch.full(
            (frames,),
            1. / states,
            dtype=observation.dtype,
            device=device)
    if transition is None:
        transition = torch.full(
            (frames, frames),
            1. / states,
            dtype=observation.dtype,
            device=device)
    
    # Forward pass
    posterior, memory = forward(observation, transition, initial)

    # Backward pass
    return backward(posterior, memory)


###############################################################################
# Individual steps
###############################################################################


def backward(posterior, memory):
    """Get optimal pass from results of forward pass"""
    # Initialize
    indices = torch.full(
        (posterior.shape[0],),
        torch.argmax(posterior[-1]),
        dtype=torch.int,
        device=posterior.device)
        
    # Backward
    for t in range(indices.shape[0] - 2, -1, -1):
        indices[t] = memory[t + 1, indices[t + 1]]
    
    return indices


def forward(observation, transition, initial):
    """Viterbi decoding forward pass"""
    # Initialize
    posterior = torch.zeros_like(observation)
    memory = torch.zeros(
        observation.shape,
        dtype=torch.int,
        device=observation.device)
    
    # Add prior to first frame
    posterior[0] = observation[0] + initial

    # Forward pass
    for t in range(1, observation.shape[0]):
        step(t, observation, transition, posterior, memory)

    return posterior, memory


def step(index, observation, transition, posterior, memory):
    """One step of the forward pass"""
    probability = posterior[index - 1] + transition

    # Update best so far
    for j in range(observation.shape[1]):
        memory[index, j] = torch.argmax(probability[j])
        posterior[index, j] = \
            observation[index, j] + probability[j, memory[index][j]]
