import numpy as np
import torch 
from torch.utils import data


class LungCancerDataset(data.Dataset):

    def __init__(self, data_table, label_table):
        super().__init__()

        self.data_table = data_table.astype('f')
        self.label_table = label_table.astype('i')

    def __len__(self):
        return len(self.label_table)

    def __getitem__(self, i):
        data = self.data_table[i, ...]
        data = data[np.newaxis, ...].astype(np.float32)
        label = np.array(self.label_table[i])

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return data, label
