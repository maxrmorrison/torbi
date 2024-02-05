import torch


###############################################################################
# Dataloader
###############################################################################


def collate(batch):
    observations, input_files = zip(*batch)

    # Handle chunking
    if isinstance(observations[0], list):
        batch_chunks = [len(obs) for obs in observations]
        observations = sum(observations, [])
    else:
        batch_chunks = [1] * len(observations)
    batch_frames = torch.tensor([obs.shape[0] for obs in observations])

    batch = len(observations)
    if batch == 0:
        raise ValueError('batch must contain at least 1 item')

    max_frames = max(observation.shape[0] for observation in observations)

    observation = torch.zeros(
        (batch, max_frames, observations[0].shape[-1]),
        dtype=observations[0].dtype)

    for i, obs in enumerate(observations):
        observation[i, :obs.shape[0]] = obs

    return observation, batch_frames, batch_chunks, input_files


def separate(indices, batch_chunks, batch_frames):
    start = 0
    separated = []
    for chunks in batch_chunks:
        frames = batch_frames[start:start+chunks]
        separated.append(
            torch.cat([
                indices[start + i, :frames[i]] for i in range(0, chunks)]))
        start += chunks
    return separated
