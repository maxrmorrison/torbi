import torch
import torbi


###############################################################################
# Dataset
###############################################################################


class Dataset(torch.utils.data.Dataset):

    def __init__(self, input_files):
        self.input_files = input_files

    def __getitem__(self, index):
        """Retrieve the indexth item"""
        input_file = self.input_files[index]

        observation = torch.load(input_file, map_location='cpu')

        # Maybe chunk observations
        if torbi.MIN_CHUNK_SIZE is not None:
            observation = torbi.chunk(observation)

        return observation, input_file

    def __len__(self):
        """Length of the dataset"""
        return len(self.input_files)
