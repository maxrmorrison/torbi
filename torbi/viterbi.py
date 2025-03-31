import torch
from typing import Optional


def decode(
    observation: torch.Tensor,
    batch_frames: torch.Tensor,
    transition: torch.Tensor,
    initial: torch.Tensor,
    num_threads: int = 0,
) -> torch.Tensor:
    """Decode a time-varying categorical distribution

    Args:
        observation: :math:`(N, T, S)` or :math:`(T, S)`
            where `S = the number of states`,
            `T = the length of the sequence`,
            and `N = batch size`.
            Time-varying categorical distribution
        batch_frames :math:`(N)`
            Sequence length of each batch item
        transition :math:`(S, S)`
            Categorical transition matrix
        initial :math:`(S)`
            Categorical initial distribution

    Return:
        indices: :math:`(N, T)`
            The decoded bin indices

    Example::

            >>> observation = torch.tensor([[
            >>>     [0.25, 0.5, 0.25],
            >>>     [0.25, 0.25, 0.5],
            >>>     [0.33, 0.33, 0.33]
            >>> ]])
            >>> batch_frames = torch.tensor([3])
            >>> transition = torch.tensor([
            >>>     [0.5, 0.25, 0.25],
            >>>     [0.33, 0.34, 0.33],
            >>>     [0.25, 0.25, 0.5]
            >>> ])
            >>> initial = torch.tensor([0.4, 0.35, 0.25])
            >>> bins = viterbi.decode(
            >>>     observation,
            >>>     batch_frames,
            >>>     transition,
            >>>     initial)
    """
    if observation.device.type == 'cpu':
        torch.set_num_threads(num_threads)
    return torch.ops.torbi.viterbi_decode(observation, batch_frames, transition, initial)
