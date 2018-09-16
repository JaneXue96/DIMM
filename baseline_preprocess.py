import pandas as pd
import numpy as np
import os
import pickle as pkl
import json
from tqdm import tqdm


def convert_samples(samples):
    print('Converting samples...')
    indexes = samples[0]['index']
    labels = [samples[0]['label']] * len(indexes)
    del samples[0]
    for sample in tqdm(samples):
        # indexes = np.vstack((indexes, sample['index']))
        indexes = np.concatenate((indexes, sample['index']), axis=0)
        labels += [sample['label']] * len(sample['index'])

    labels = np.asarray(labels)
    return indexes, labels


def preprocess_data(data_path, task_type):
    samples = []
    labels = []
    total = 0
    print('Reading raw files for {}...'.format(task_type))
    for file in tqdm(os.listdir(data_path)):
        total += 1
        if file.startswith('0'):
            dead = 0
        else:
            dead = 1
        raw_sample = pd.read_csv(os.path.join(data_path, file), sep=',')
        raw_sample = raw_sample.fillna(0)
        medicine = raw_sample.iloc[:, 209:].as_matrix()
        index = raw_sample.iloc[:, 3:208].as_matrix()
        index = np.concatenate((index, medicine), axis=1)
        index = index.tolist()
        samples += index
        labels += [dead] * len(index)
        # sample = {'patient_id': total,
        #           'index': index,
        #           'label': dead,
        #           'name': file}
        # samples.append(sample)
    # train_samples, test_samples = train_test_split(samples, test_size=0.2)
    # dim = samples[0]['index'].shape[1]
    dim = len(samples[0])
    # del samples
    # indexes, labels = convert_samples(samples)
    print('Num of samples : ', len(samples))
    return [np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int32)], dim


def save(data, file_name, data_type):
    print('Saving {} data..c.'.format(data_type))
    for d in data:
        np.save(file_name, d)
    # with open(file_name, 'w') as f:
    #     json.dump(data, f)
    # f.close()


single_task = ['5849', '25000', '41401', '4019']
for path in single_task:
    train_data, dim = preprocess_data('data/raw_data/' + path + '/train', path)
    save(train_data, 'data/preprocessed_data/baseline/' + path + '/train.npy', 'train')
    # test_data, dim = preprocess_data('data/raw_data/' + path + '/test', path)
    # save(test_data, 'data/preprocessed_data/baseline/' + path + '/test.npy', 'test')
