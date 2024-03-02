from torch import Tensor
from typing import Optional

def decode(
    observation: Tensor,
    batch_frames: Tensor,
    transition: Tensor,
    initial: Tensor,
    num_threads: Optional[int] = 0
):
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
        num_threads (int, optional)
            Number of threads to use if doing CPU decoding

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
            >>> bins = viterbi.decode(observation, batch_frames, transition, initial)
    """
