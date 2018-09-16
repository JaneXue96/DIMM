import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc


def get_record_parser(max_len, dim):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'patient_id': tf.FixedLenFeature([], tf.int64),
                                               'index': tf.FixedLenFeature([], tf.string),
                                               'medicine': tf.FixedLenFeature([], tf.string),
                                               'seq_len': tf.FixedLenFeature([], tf.int64),
                                               'label': tf.FixedLenFeature([], tf.int64)
                                           })
        index = tf.reshape(tf.decode_raw(features['index'], tf.float32), [max_len, dim[0]])
        medicine = tf.reshape(tf.decode_raw(features['medicine'], tf.float32), [max_len, dim[1]])
        label = tf.to_int32(features['label'])
        seq_len = tf.to_int32(features['seq_len'])
        patient_id = features['patient_id']
        return patient_id, index, medicine, seq_len, label
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        config.capacity).batch(config.train_batch).repeat()

    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).batch(config.dev_batch).repeat(config.epochs)

    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle, is_point, logger):
    losses = []
    pre_labels, ref_labels = [], []
    metrics = {}
    pre_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    ref_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    for _ in range(num_batches):
        patient_ids, loss, labels, seq_lens = sess.run([model.id, model.loss, model.pre_labels, model.seq_len],
                                                       feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        for pid, pre_label, seq_len in zip(patient_ids, labels, seq_lens):
            sample = eval_file[str(pid)]
            if is_point:
                pre_labels.append(pre_label)
                ref_labels.append(sample['label'])
            else:
                ref_label = np.zeros([seq_len], dtype=np.int32)
                ref_label[:] = sample['label']
                ref_labels += [sample['label']] * seq_len
                pre_labels += pre_label[:seq_len].tolist()

            for k, v in pre_points.items():
                if seq_len >= k:
                    v.append(pre_label[k - 1])
                    ref_points[k].append(sample['label'])

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(ref_labels, pre_labels)
    metrics['roc'] = roc_auc_score(ref_labels, pre_labels)
    (precisions, recalls, thresholds) = precision_recall_curve(ref_labels, pre_labels)
    metrics['prc'] = auc(recalls, precisions)
    metrics['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    for k, v in pre_points.items():
        logger.info('{} hour confusion matrix. AUCROC : {}'.format(int(k / 3), roc_auc_score(ref_points[k], v)))
        logger.info(confusion_matrix(ref_points[k], v))
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(ref_labels, pre_labels))
    return metrics


def multi_evaluate(model, num_batches, eval_file, sess, handle, str_handle, is_point):
    losses = []
    task_metrics = {}
    task_labels = {}
    tasks = ['5849', '25000']
    for t in tasks:
        task_metrics[t] = {'loss': 0, 'acc': 0.0, 'roc': 0.0, 'prc': 0.0, 'pse': 0.0}
        task_labels[t] = {'true': [], 'pred': []}
    for _ in range(num_batches):
        patient_ids, loss, labels, seq_lens = sess.run([model.id, model.loss, model.pre_labels, model.seq_len],
                                                       feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        for pid, pre_label, seq_len in zip(patient_ids, labels, seq_lens):
            sample = eval_file[str(pid)]
            task = sample['task']
            if is_point:
                task_labels[task]['pred'].append(pre_label)
                task_labels[task]['true'].append(sample['label'])
            else:
                task_labels[task]['true'] += [sample['label']] * seq_len
                task_labels[task]['pred'] += pre_label[:seq_len].tolist()
            # for k, v in pre_points.items():
            #     if seq_len >= k:
            #         v.append(pre_label[k - 1])
            #         ref_points[k].append(sample['label'])
            # ref_score = sample['score']
            # mses.append(mean_squared_error(sample['score'][:seq_len], pre_score[:seq_len]))
    for t in tasks:
        # task_metrics[t]['loss'] = np.mean(losses)
        task_metrics[t]['acc'] = accuracy_score(task_labels[t]['true'], task_labels[t]['pred'])
        task_metrics[t]['roc'] = roc_auc_score(task_labels[t]['true'], task_labels[t]['pred'])
        (precisions, recalls, thresholds) = precision_recall_curve(task_labels[t]['true'], task_labels[t]['pred'])
        task_metrics[t]['prc'] = auc(recalls, precisions)
        task_metrics[t]['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return task_metrics
    # loss_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=metrics['loss']), ])
    # acc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=metrics['acc']), ])
    # auc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/roc'.format(data_type), simple_value=metrics['roc']), ])
    # prc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/prc'.format(data_type), simple_value=metrics['prc']), ])
    # return metrics, (loss_sum, acc_sum, auc_sum, prc_sum)
