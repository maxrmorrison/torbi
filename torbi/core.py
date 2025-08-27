import contextlib
import math
import multiprocessing as mp
import os
from typing import List, Optional, Union, Dict

import numpy as np
import torch
import torchutil

import torbi
from torbi import decode


###############################################################################
# Viterbi decoding
###############################################################################


# def from_dataloader(
#     dataloader: torch.utils.data.DataLoader,
#     output_files: Dict[
#         Union[str, bytes, os.PathLike],
#         Union[str, bytes, os.PathLike]],
#     transition: Optional[torch.Tensor] = None,
#     initial: Optional[torch.Tensor] = None,
#     log_probs: bool = False,
#     save_workers: int = 0,
#     gpu: Optional[int] = None,
#     num_threads: Optional[int] = None
# ) -> None:
#     """Decode time-varying categorical distributions from dataloader

#     Arguments
#         dataloader
#             torbi DataLoader for asynchronous loading and preprocessing
#         output_files
#             A dictionary mapping input filenames to output filenames
#         transition
#             Categorical transition matrix; defaults to uniform
#             shape=(states, states)
#         initial
#             Categorical initial distribution; defaults to uniform
#             shape=(states,)
#         log_probs
#             Whether inputs are in (natural) log space
#         save_workers
#             The number of worker threads to use for async file saving
#         gpu
#             The index of the GPU to use for inference
#         num_threads
#             The number of threads to use for parallelized decoding

#     Returns
#         indices
#             The decoded bin indices
#             shape=(batch, frames)
#     """
#     # Setup progress bar
#     progress = torchutil.iterator(
#         range(0, len(dataloader.dataset)),
#         torbi.CONFIG,
#         total=len(dataloader.dataset))

#     # Iterate over dataset
#     for observation, batch_frames, batch_chunks, input_filenames in dataloader:

#         # Decode
#         indices = from_probabilities(
#             observation=observation,
#             batch_frames=batch_frames,
#             transition=transition,
#             initial=initial,
#             log_probs=log_probs,
#             gpu=gpu,
#             num_threads=num_threads)

#         # Get output filenames
#         filenames = [output_files[file] for file in input_filenames]

#         if torbi.USE_CHUNKING:

#             # Combine chunks
#             indices = torbi.data.separate(
#                 indices=indices,
#                 batch_chunks=batch_chunks,
#                 batch_frames=batch_frames)

#             # Save
#             for indices, filename in zip(indices, filenames):
#                 save(indices.cpu().detach(), filename)

#         else:

#             # Save
#             for indices, filename, frames in zip(
#                 indices.cpu().detach(),
#                 filenames,
#                 batch_frames.cpu()
#             ):
#                 save_masked(indices, filename, frames)

#         # Increment by batch size
#         progress.update(len(input_filenames))

#     # Close progress bar
#     progress.close()


def from_probabilities(
    observation: torch.Tensor,
    batch_frames: Optional[torch.Tensor] = None,
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = 1
) -> torch.Tensor:
    """Decode a time-varying categorical distribution

    Arguments
        observation
            Time-varying categorical distribution
            shape=(batch, frames, states)
        batch_frames
            Number of frames in each batch item; defaults to all
            shape=(batch,)
        transition
            Categorical transition matrix; defaults to uniform
            shape=(states, states)
        initial
            Categorical initial distribution; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding

    Returns
        indices
            The decoded bin indices
            shape=(batch, frames)
    """
    batch, frames, states = observation.shape
    if gpu == 'mps':
        device = torch.device('mps')
    else:
        device = 'cpu' if gpu is None else f'cuda:{gpu}'

    if batch_frames is None:
        batch_frames = torch.full(
            (batch,),
            frames,
            dtype=torch.int32,
            device=device)
    batch_frames = batch_frames.to(dtype=torch.int32, device=device)

    # Default to uniform initial probabilities
    if initial is None:
        initial = torch.full(
            (states,),
            math.log((1. / states) + torch.finfo(torch.float32).tiny),
            dtype=torch.float32,
            device=device)

    # Ensure initial probabilities are in log space
    else:
        if not log_probs:
            initial = torch.log(initial)
        initial = initial.to(device)

    # Default to uniform transition probabilities
    if transition is None:
        transition = torch.full(
            (states, states),
            math.log(1. / states),
            dtype=torch.float32,
            device=device)

    # Ensure transition probabilities are in log space
    else:
        if not log_probs:
            transition = torch.log(transition)
        transition = transition.to(device)

    # Ensure observation probabilities are in log space
    if not log_probs:
        observation = torch.log(observation)
    observation = observation.to(device=device, dtype=torch.float32)

    # Add epsilon for stability
    # NOTE - may break gradients
    torch.exp_(observation)
    observation += torch.finfo(torch.float32).tiny
    torch.log_(observation)

    # Decode
    with torchutil.time.context('torbi'):
        indices = decode(
            observation,
            batch_frames,
            transition,
            initial,
            num_threads=num_threads)

    return indices


def from_file(
    input_file: Union[str, os.PathLike],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = 1
) -> torch.Tensor:
    """Decode a time-varying categorical distribution file

    Arguments
        input_file
            Time-varying categorical distribution file
            shape=(frames, states)
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding

    Returns
        indices
            The decoded bin indices
            shape=(frames,)
    """
    # Load observations
    observation = torch.load(input_file).unsqueeze(dim=0)

    # Load transition probabilities
    if transition_file:
        transition = torch.load(transition_file)
        if log_probs:
            transition = torch.log(transition)
    else:
        transition = None

    # Load initial distribution
    if initial_file:
        initial = torch.load(initial_file)
    else:
        initial = None

    # Decode
    return from_probabilities(
        observation=observation,
        transition=transition,
        initial=initial,
        log_probs=log_probs,
        gpu=gpu,
        num_threads=num_threads)


def from_file_to_file(
    input_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = None
) -> None:
    """Decode a time-varying categorical distribution file and save

    Arguments
        input_file
            Time-varying categorical distribution file
            shape=(frames, states)
        output_file
            File to save decoded indices
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding
    """
    indices = from_file(
        input_file,
        transition_file,
        initial_file,
        log_probs,
        gpu=gpu,
        num_threads=num_threads)
    torch.save(indices, output_file)


def from_files_to_files(
    input_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = None
) -> None:
    """Decode time-varying categorical distribution files and save

    Arguments
        input_files
            Time-varying categorical distribution files
            shape=(frames, states)
        output_files
            Files to save decoded indices
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding
    """
    # Load transition probabilities
    if transition_file:
        transition = torch.load(transition_file)
        if log_probs:
            transition = torch.log(
                transition + torch.finfo(transition.dtype).tiny)
    else:
        transition = None

    # Load initial distribution
    if initial_file:
        initial = torch.load(initial_file)
    else:
        initial = None

    # Preserve file mapping
    mapping = {
        input_file: output_file
        for input_file, output_file in zip(input_files, output_files)}

    # Decode
    from_dataloader(
        dataloader=torbi.data.loader(input_files),
        output_files=mapping,
        transition=transition,
        initial=initial,
        log_probs=log_probs,
        gpu=gpu,
        num_threads=num_threads)


###############################################################################
# Utilities
###############################################################################


def from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    output_files: Dict[
        Union[str, bytes, os.PathLike],
        Union[str, bytes, os.PathLike]],
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = 1
) -> None:
    """Decode time-varying categorical distributions from dataloader

    Arguments
        dataloader
            torbi DataLoader object for asynchronous loading and preprocessing
        output_files
            A dictionary mapping input filenames to output filenames
        transition
            Categorical transition matrix; defaults to uniform
            shape=(states, states)
        initial
            Categorical initial distribution; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            The index of the GPU to use for inference

    Returns
        indices
            The decoded bin indices
            shape=(batch, frames)
    """
    # Setup progress bar
    progress = torchutil.iterator(
        range(0, len(dataloader.dataset)),
        torbi.CONFIG,
        total=len(dataloader.dataset))

    # Iterate over dataset
    for (
        observation,
        batch_frames,
        batch_chunks,
        input_filenames
    ) in dataloader:

        # Decode
        indices = from_probabilities(
            observation=observation,
            batch_frames=batch_frames,
            transition=transition,
            initial=initial,
            log_probs=log_probs,
            gpu=gpu,
            num_threads=num_threads)

        # Get output filenames
        filenames = [output_files[file] for file in input_filenames]

        # Save
        if torbi.MIN_CHUNK_SIZE is not None:

            # Combine chunks
            indices = torbi.data.separate(
                indices=indices,
                batch_chunks=batch_chunks,
                batch_frames=batch_frames)

            # Save
            for indices, filename in zip(indices, filenames):
                save(indices.cpu().detach(), filename)
        else:

            # Save
            for indices, filename, frames in zip(
                indices.cpu().detach(),
                filenames,
                batch_frames.cpu()
            ):
                save_masked(indices, filename, frames)

        # Increment by batch size
        progress.update(len(input_filenames))

    # Close progress bar
    progress.close()


def save(tensor, file):
    """Save tensor"""
    torch.save(tensor.clone(), file)


def save_masked(tensor, file, length):
    """Save masked tensor"""
    torch.save(tensor[..., :length].clone(), file)
