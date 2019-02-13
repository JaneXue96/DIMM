import pandas as pd
import numpy as np
from scipy import stats
import os
import ujson as json
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


def divide_data(train_data, test_data):
    train_samples, test_samples = [], []
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
        sample = {'patient_id': total,
                  'index': index,
                  'medicine': medicine,
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
        sample = {'patient_id': total,
                  'index': index,
                  'medicine': medicine,
                  'label': dead,
                  'name': file}
        test_samples.append(sample)

    index_dim = train_samples[0]['index'].shape[1]
    medicine_dim = train_samples[0]['medicine'].shape[1]
    train_eval_samples = {}
    for sample in train_samples:
        train_eval_samples[str(sample['patient_id'])] = {'label': sample['label'],
                                                         'name': sample['name']}

    test_eval_samples = {}
    for sample in test_samples:
        test_eval_samples[str(sample['patient_id'])] = {'label': sample['label'],
                                                        'name': sample['name']}

    return train_samples, test_samples, train_eval_samples, test_eval_samples, max_len, (index_dim, medicine_dim)


def preprocess_data(data_path):
    samples = []
    total = 0
    max_len = 0
    seq_len = []
    dead_len, live_len = 0, 0
    # scores = []
    print('Reading raw files...')
    for file in tqdm(os.listdir(data_path)):
        total += 1
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        # try:
        raw_sample = pd.read_csv(os.path.join(data_path, file), sep=',')
        # except:
        # print(file)
        raw_sample = raw_sample.fillna(0)
        medicine = raw_sample.iloc[:, 209:].as_matrix()
        index = raw_sample.iloc[:, 3:208].as_matrix()
        # score = raw_sample['totalScore'].values.tolist()
        # for i, idx in enumerate(index):
        #     if not np.all(idx == np.array(list(idx))):
        #         print(file)
        #         break
        length = index.shape[0]
        if length > max_len:
            max_len = length
        if length == 0:
            print(file)
        sample = {'patient_id': total,
                  'index': index,
                  'medicine': medicine,
                  # 'score': score,
                  'label': dead,
                  'name': file}
        samples.append(sample)
        # scores.append(np.mean(score))
        seq_len.append(length)
        if dead == 0:
            dead_len += length
        else:
            live_len += length

    # print(stats.describe(np.asarray(scores)))
    # stat(seq_len)
    print('Dead length {}'.format(dead_len))
    print('Live length {}'.format(live_len))
    train_samples, test_samples = train_test_split(samples, test_size=0.2)
    index_dim = samples[0]['index'].shape[1]
    medicine_dim = samples[0]['medicine'].shape[1]
    del samples
    train_eval_samples = {}
    for sample in train_samples:
        train_eval_samples[str(sample['patient_id'])] = {'label': sample['label'], 'name': sample['name']}
    test_eval_samples = {}
    for sample in test_samples:
        test_eval_samples[str(sample['patient_id'])] = {'label': sample['label'], 'name': sample['name']}

    return train_samples, test_samples, train_eval_samples, test_eval_samples, max_len, (index_dim, medicine_dim)


def scale_data(data_path):
    X = []
    Y = []
    total = 0
    max_len = 0
    print('Reading raw files...')
    for file in tqdm(os.listdir(data_path)):
        total += 1
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        raw_sample = pd.read_csv(os.path.join(data_path, file), sep=',')
        raw_sample = raw_sample.fillna(0)
        medicine = raw_sample.iloc[:, 103:].as_matrix()
        index = raw_sample.iloc[:, 4:102].as_matrix()
        score = raw_sample['totalScore'].values.tolist()

        length = index.shape[0]
        if length > max_len:
            max_len = length

        X.append((index, medicine))
        Y.append((total, score, dead, file))

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    index_dim = X[0][0].shape[1]
    medicine_dim = X[0][1].shape[1]
    del X, Y
    train_index, train_medicine = stack_index(train_X, 'train')
    del train_X
    test_index, test_medicine = stack_index(test_X, 'test')
    del test_X
    # scaler = MinMaxScaler()
    index_scaler = StandardScaler()
    train_index = index_scaler.fit_transform(train_index)
    test_index = index_scaler.transform(test_index)
    medicine_scaler = StandardScaler()
    train_medicine = medicine_scaler.fit_transform(train_medicine)
    test_medicine = medicine_scaler.fit_transform(test_medicine)

    train_samples, train_eval_samples = merge_samples(train_index, train_medicine, train_Y, 'train')
    del train_index, train_medicine, train_Y
    test_samples, test_eval_samples = merge_samples(test_index, test_medicine, test_Y, 'test')
    del test_index, test_medicine, test_Y

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
        # score = np.zeros([max_len], dtype=np.int32)
        # label = np.zeros([max_len], dtype=np.int32)

        seq_len = min(len(sample['index']), max_len)
        index[:seq_len] = sample['index'][:seq_len]
        medicine[:seq_len] = sample['medicine'][:seq_len]
        # score[:seq_len] = sample['score'][:seq_len]
        # label[:seq_len] = sample['label']

        record = tf.train.Example(features=tf.train.Features(feature={
            'patient_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['patient_id']])),
            'index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[index.tostring()])),
            'medicine': tf.train.Feature(bytes_list=tf.train.BytesList(value=[medicine.tostring()])),
            'seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
            # 'score': tf.train.Feature(bytes_list=tf.train.BytesList(value=[score.tostring()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[sample['label']])),
        }))
        writer.write(record.SerializeToString())
    print('Build {} instances of features in total'.format(total))
    meta['total'] = total
    writer.close()
    return meta


def run_prepare(config, flags):
    # train_samples, test_samples, train_eval_samples, test_eval_samples, max_len, dim = preprocess_data(config.raw_dir)
    train_samples, test_samples, train_eval_samples, test_eval_samples, max_len, dim = divide_data(
        config.raw_dir + '/train',
        config.raw_dir + '/test')

    train_meta = build_features(train_samples, 'train', config.max_len, dim, flags.train_record_file)
    save(flags.train_eval_file, train_eval_samples, message='train eval')
    save(flags.train_meta, train_meta, message='train meta')
    del train_samples, train_eval_samples, train_meta

    dev_meta = build_features(test_samples, 'dev', config.max_len, dim, flags.dev_record_file)
    save(flags.dev_eval_file, test_eval_samples, message='dev eval')
    save(flags.dev_meta, dev_meta, message='dev meta')
    del test_samples, test_eval_samples, dev_meta

    return max_len, dim
