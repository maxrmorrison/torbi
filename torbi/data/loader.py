import torch

import torbi

###############################################################################
# Dataloader
###############################################################################


def loader(
    input_files,
    num_workers=torbi.NUM_WORKERS):
    """Retrieve a data loader"""
    # Initialize dataset
    dataset = torbi.data.Dataset(input_files)

    # Initialize dataloader
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=torbi.BATCH_SIZE,
        shuffle=False,
        collate_fn=torbi.data.collate)