import pandas as pd
import numpy as np
from scipy import stats
import os
from tqdm import tqdm
import pickle as pkl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def stat(seq_length):
    print('Seq len info :')
    seq_len = np.asarray(seq_length)
    idx = np.arange(0, len(seq_len), dtype=np.int32)
    print(stats.describe(seq_len))
    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.plot(idx[:], seq_len[:], 'ro')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel('seq_len')
    plt.title('Scatter Plot')

    plt.subplot(122)
    plt.hist(seq_len, bins=10, label=['seq_len'])
    plt.grid(True)
    plt.xlabel('seq_len')
    plt.ylabel('freq')
    plt.title('Histogram')
    plt.savefig('./seq_len_stats.jpg', format='jpg')


def preprocess_data(data_path):
    samples, seq_len = [], []
    max_len, dead_len, live_len = 0, 0, 0
    meta = {}
    print('Reading raw files...')
    for file in tqdm(os.listdir(data_path)):
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        raw_sample = pd.read_csv(os.path.join(data_path, file), sep=',')
        raw_sample = raw_sample.fillna(0)
        # columns_size = raw_sample.columns.size CW101 231 360 NZ390.astype(object)
        medicine = raw_sample.iloc[:, 209:].as_matrix()
        index = raw_sample.iloc[:, 3:208].as_matrix()
        # raw_sample.drop(raw_sample.columns[medicine], axis=1, inplace=True)

        # for i, idx in enumerate(index):
        #     if not np.all(idx == np.array(list(idx))):
        #         print(file)
        #         break
        length = index.shape[0]
        if length > max_len:
            max_len = length
        sample = {'index': index,
                  'medicine': medicine,
                  'length': length,
                  'label': dead,
                  'name': file}
        samples.append(sample)
        seq_len.append(length)
        if dead == 0:
            dead_len += length
        else:
            live_len += length
    stat(seq_len)
    print('Dead length {}'.format(dead_len))
    print('Live length {}'.format(live_len))
    train_samples, test_samples = train_test_split(samples, test_size=0.2)
    del samples
    meta['train_total'] = len(train_samples)
    meta['test_total'] = len(test_samples)
    index_dim = train_samples[0]['index'].shape[1]
    medicine_dim = train_samples[0]['medicine'].shape[1]
    print('Train total {} Test total {}'.format(meta['train_total'], meta['test_total']))
    print('Index dim {} Medicine dim {}'.format(index_dim, medicine_dim))

    return train_samples, test_samples, max_len, meta, (index_dim, medicine_dim)


def divide_data(train_data, test_data):
    train_samples, test_samples = [], []
    meta = {}
    total = 0
    max_len = 0
    print('Reading raw files...')
    for file in tqdm(os.listdir(train_data)):
        total += 1
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        raw_sample = pd.read_csv(os.path.join(train_data, file), sep=',')
        raw_sample = raw_sample.fillna(0)
        medicine = raw_sample.iloc[:, 209:].as_matrix()
        index = raw_sample.iloc[:, 3:208].as_matrix()
        length = index.shape[0]
        if length > max_len:
            max_len = length
        sample = {'index': index,
                  'medicine': medicine,
                  'length': length,
                  'label': dead,
                  'name': file}
        train_samples.append(sample)

    for file in tqdm(os.listdir(test_data)):
        total += 1
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        raw_sample = pd.read_csv(os.path.join(test_data, file), sep=',')
        raw_sample = raw_sample.fillna(0)
        medicine = raw_sample.iloc[:, 209:].as_matrix()
        index = raw_sample.iloc[:, 3:208].as_matrix()
        length = index.shape[0]
        if length > max_len:
            max_len = length
        sample = {'index': index,
                  'medicine': medicine,
                  'length': length,
                  'label': dead}
        test_samples.append(sample)

    index_dim = train_samples[0]['index'].shape[1]
    medicine_dim = train_samples[0]['medicine'].shape[1]
    meta['train_total'] = len(train_samples)
    meta['test_total'] = len(test_samples)
    # train_eval_samples = {}
    # for sample in train_samples:
    #     train_eval_samples[str(sample['patient_id'])] = {'label': sample['label']}
    #
    # test_eval_samples = {}
    # for sample in test_samples:
    #     test_eval_samples[str(sample['patient_id'])] = {'label': sample['label']}

    return train_samples, test_samples, max_len, meta, (index_dim, medicine_dim)


def save(filename, obj, message=None):
    if message is not None:
        print('Saving {}...'.format(message))
        with open(filename, 'wb') as fh:
            pkl.dump(obj, fh)


def run_prepare(config, flags):
    # train_samples, dev_samples, max_len, meta, dim = preprocess_data(config.raw_dir)
    train_samples, dev_samples, max_len, meta, dim = divide_data(config.raw_dir + '/train',
                                                                 config.raw_dir + '/test')
    save(flags.train_file, train_samples, message='train file')
    del train_samples
    save(flags.eval_file, dev_samples, message='eval file')
    save(flags.meta, meta, message='meta file')
    del dev_samples

    return max_len, dim
