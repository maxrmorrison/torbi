import math

import librosa
import numpy as np
import torch
import torchutil


###############################################################################
# Reference implementation of Viterbi decoding
###############################################################################


def from_probabilities(
    observation,
    transition=None,
    initial=None,
    log_probs=None
) -> torch.Tensor:
    """Perform reference Viterbi decoding"""
    device = observation.device
    frames, states = observation.shape

    # Setup initial probabilities
    if initial is None:
        initial = np.full(
            (frames,),
            math.log(1. / states),
            dtype=np.float32)
    else:
        if log_probs:
            initial = torch.exp(initial)
        initial = initial.cpu().numpy().astype(np.float32)

    # Setup transition probabilities
    if transition is None:
        transition = np.full(
            (frames, frames),
            math.log(1. / states),
            dtype=np.float32)
    else:
        if log_probs:
            transition = torch.exp(transition)
        transition = transition.cpu().numpy().astype(np.float32)

    # Setup observation probabilities
    if log_probs:
        observation = torch.exp(observation)
    observation = observation.cpu().numpy().astype(np.float32)

    # Decode
    with torchutil.time.context('librosa'):
        indices = librosa.sequence.viterbi(
            observation.T,
            transition,
            p_init=initial)

    # Cast to torch
    return torch.tensor(
        indices.astype(np.int32),
        dtype=torch.int,
        device=device)


def from_file(
    input_file,
    transition=None,
    initial=None,
    log_probs=False
) -> torch.Tensor:
    """Perform reference Viterbi decoding on a file"""
    observation = torch.load(input_file)
    return from_probabilities(observation, transition, initial, log_probs)


def from_file_to_file(
    input_file,
    output_file,
    transition=None,
    initial=None,
    log_probs=False
) -> None:
    """Perform reference Viterbi decoding on a file and save"""
    indices = from_file(input_file, transition, intitial, log_probs)
    torch.save(indices, output_file)


def from_files_to_files(
    input_files,
    output_files,
    transition=None,
    initial=None,
    log_probs=False
) -> None:
    """Perform reference Viterbi decoding on many files and save"""
    for input_file, output_file in zip(input_files, output_files):
        from_file_to_file(input_file, output_file)
