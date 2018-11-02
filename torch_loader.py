import torch
import torch.utils.data
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def pad_tensor(vec, pad, dim, vec_type):
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
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec_type)], dim=dim)


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
        padded_batch = list(map(lambda x_y: (pad_tensor(x_y[0], max_len, self.dim, torch.float),
                                             pad_tensor(x_y[1], max_len, self.dim, torch.float),
                                             pad_tensor(x_y[2], max_len, self.dim, torch.long),
                                             x_y[3]),
                                batch))
        # stack all
        xs = torch.stack(tuple(map(lambda x: x[0], padded_batch)), dim=0)
        ys = torch.stack(tuple(map(lambda x: x[1], padded_batch)), dim=0)
        zs = torch.stack(tuple(map(lambda x: x[2], padded_batch)), dim=0)
        ls = torch.LongTensor(tuple(map(lambda x: x[3], padded_batch)))
        return xs, ys, zs, ls

    def __call__(self, batch):
        return self.pad_collate(batch)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path_npy):
        self.raw_file = np.load(path_npy)

    def __getitem__(self, index):
        file_path = self.raw_file[index]
        raw_sample = pd.read_csv(file_path, sep=',')
        raw_sample = raw_sample.fillna(0)
        measure = raw_sample.iloc[:, 3:208].as_matrix().astype(np.float32)
        treat = raw_sample.iloc[:, 209:].as_matrix().astype(np.float32)
        tag = 1 if int(file_path.split('/')[-1][0]) == 0 else 0
        label = np.array([tag] * len(treat), dtype=np.int64)
        seq_len = len(measure)
        return torch.from_numpy(measure), torch.from_numpy(treat), torch.from_numpy(label), seq_len

    def __len__(self):
        return len(self.raw_file)


def gen_path_npy(source_path, target_path, data_type):
    file_list = []
    print('Generating {} data path npy...'.format(data_type))
    for file in tqdm(os.listdir(source_path)):
        file = os.path.join(source_path, file)
        file_list.append(file)
    file_npy = np.array(file_list)
    target_path = os.path.join(target_path, data_type + '.npy')

    np.save(target_path, file_npy)


def run_prepare(config):
    data_type = ['train', 'test']
    for t in data_type:
        gen_path_npy(os.path.join(config.raw_dir, t), config.preprocessed_dir, t)
