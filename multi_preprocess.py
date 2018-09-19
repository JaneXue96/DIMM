import pandas as pd
import numpy as np
from scipy import stats
import os
import ujson as json
from tqdm import tqdm
import tensorflow as tf
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
    # plt.show()


def stack_index(samples, data_type):
    print('Stacking {} samples...'.format(data_type))
    indexes = [sample[0] for sample in samples]
    medicines = [sample[1] for sample in samples]
    return np.concatenate(indexes, axis=0), np.concatenate(medicines, axis=0)


def merge_samples(scaled_indexes, scaled_medicine, packed, data_type):
    samples = []
    eval_samples = {}
    start, end = 0, 0
    print('Merging {} samples...'.format(data_type))
    for pack in tqdm(packed):
        end = start + len(pack[1])
        samples.append({'patient_id': pack[0],
                        'index': scaled_indexes[start:end],
                        'medicine': scaled_medicine[start:end],
                        'score': pack[1],
                        'label': pack[2],
                        'name': pack[3]})
        eval_samples[str(pack[0])] = {'score': pack[1], 'label': pack[2], 'name': pack[3]}

    print('Got {} {} samples.'.format(len(samples), data_type))
    return samples, eval_samples


def preprocess_data(data_path):
    train_samples, test_samples = [], []
    total = 0
    max_len = 0
    seq_len = []
    dead_len, live_len = 0, 0
    tasks = ['5849', '25000', '41401', '4019']
    for t in tasks:
        print('Reading raw files of task ' + t)
        path = os.path.join(data_path, t)
        train_path = os.path.join(path, 'train')
        for file in tqdm(os.listdir(train_path)):
            total += 1
            if file.startswith('0'):
                dead = 0
            else:
                dead = 1
            raw_sample = pd.read_csv(os.path.join(train_path, file), sep=',')
            raw_sample = raw_sample.fillna(0)
            medicine = raw_sample.iloc[:, 209:].as_matrix()
            index = raw_sample.iloc[:, 3:208].as_matrix()
            length = index.shape[0]
            if length > max_len:
                max_len = length
            sample = {'patient_id': total,
                      'index': index,
                      'medicine': medicine,
                      'label': dead}
            train_samples.append(sample)
            seq_len.append(length)
            if dead == 0:
                dead_len += length
            else:
                live_len += length
        test_path = os.path.join(path, 'test')
        for file in tqdm(os.listdir(test_path)):
            total += 1
            if file.startswith('0'):
                dead = 0
            else:
                dead = 1
            raw_sample = pd.read_csv(os.path.join(test_path, file), sep=',')
            raw_sample = raw_sample.fillna(0)
            medicine = raw_sample.iloc[:, 209:].as_matrix()
            index = raw_sample.iloc[:, 3:208].as_matrix()
            length = index.shape[0]
            if length > max_len:
                max_len = length
            sample = {'patient_id': total,
                      'index': index,
                      'medicine': medicine,
                      'label': dead,
                      'task': t}
            test_samples.append(sample)
    index_dim = train_samples[0]['index'].shape[1]
    medicine_dim = train_samples[0]['medicine'].shape[1]
    train_eval_samples = {}
    for sample in train_samples:
        train_eval_samples[str(sample['patient_id'])] = {'label': sample['label']}

    test_eval_samples = {}
    for sample in test_samples:
        test_eval_samples[str(sample['patient_id'])] = {'label': sample['label'],
                                                        'task': sample['task']}
    return train_samples, test_samples, train_eval_samples, test_eval_samples, max_len, (index_dim, medicine_dim)


def save(filename, obj, message=None):
    if message is not None:
        print('Saving {}...'.format(message))
        with open(filename, 'w') as fh:
            json.dump(obj, fh)


def build_features(samples, data_type, max_len, dim, out_file):
    print('Processing {} examples...'.format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    meta = {}
    for sample in tqdm(samples):
        total += 1
        index = np.zeros([max_len, dim[0]], dtype=np.float32)
        medicine = np.zeros([max_len, dim[1]], dtype=np.float32)

        seq_len = min(len(sample['index']), max_len)
        index[:seq_len] = sample['index'][:seq_len]
        medicine[:seq_len] = sample['medicine'][:seq_len]

        record = tf.train.Example(features=tf.train.Features(feature={
            'patient_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['patient_id']])),
            'index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[index.tostring()])),
            'medicine': tf.train.Feature(bytes_list=tf.train.BytesList(value=[medicine.tostring()])),
            'seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['label']])),
        }))
        writer.write(record.SerializeToString())
    print('Build {} instances of features in total'.format(total))
    meta['total'] = total
    writer.close()
    return meta


def run_prepare(config, flags):
    train_samples, dev_samples, train_eval_samples, dev_eval_samples, max_len, dim = preprocess_data(config.raw_dir)
    train_meta = build_features(train_samples, 'train', config.max_len, dim, flags.train_record_file)
    save(flags.train_eval_file, train_eval_samples, message='train eval')
    save(flags.train_meta, train_meta, message='train meta')
    del train_samples, train_eval_samples, train_meta

    dev_meta = build_features(dev_samples, 'dev', config.max_len, dim, flags.dev_record_file)
    save(flags.dev_eval_file, dev_eval_samples, message='dev eval')
    save(flags.dev_meta, dev_meta, message='dev meta')
    del dev_samples, dev_eval_samples, dev_meta
    return max_len, dim
