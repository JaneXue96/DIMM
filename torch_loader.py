import torch
import pandas as pd
import numpy as np


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda x_y: (pad_tensor(x_y[0], pad=max_len, dim=self.dim),
                                 pad_tensor(x_y[1], pad=max_len, dim=self.dim),
                                 pad_tensor(x_y[2], pad=max_len, dim=self.dim)),
                    batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.stack(map(lambda x: x[1], batch), dim=0)
        zs = torch.stack(map(lambda x: x[2], batch), dim=0)
        # zs = torch.LongTensor(map(lambda x: x[2], batch))
        return xs, ys, zs

    def __call__(self, batch):
        return self.pad_collate(batch)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path_npy):
        self.raw_file = path_npy

    def __getitem__(self, index):
        file_path = self.raw_file[index]
        raw_sample = pd.read_csv(file_path, sep=',')
        raw_sample = raw_sample.fillna(0)
        treat = raw_sample.iloc[:, 209:].as_matrix()
        measure = raw_sample.iloc[:, 3:208].as_matrix()
        label = np.array([int(file_path.split('/')[-1][0])]*len(treat), dtype=np.int32)
        return measure, treat, label

    def __len__(self):
        return len(self.raw_file)
