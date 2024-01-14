import torch


###############################################################################
# Dataloader
###############################################################################


def collate(batch):
    observations, batch_frames, input_files = zip(*batch)
    
    batch = len(observations)
    if batch == 0:
        raise ValueError('batch must contain at least 1 item')

    max_frames = max(observation.shape[0] for observation in observations)

    observation = torch.zeros((batch, max_frames, observations[0].shape[-1]), dtype=observations[0].dtype)

    for i, obs in enumerate(observations):
        observation[i, :obs.shape[0]] = obs

    batch_frames = torch.tensor(batch_frames)

    return (observation, batch_frames, input_files)