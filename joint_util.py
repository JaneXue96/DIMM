import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, f1_score


def get_record_parser(max_len, dim):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'patient_id': tf.FixedLenFeature([], tf.int64),
                                               'index': tf.FixedLenFeature([], tf.string),
                                               'medicine': tf.FixedLenFeature([], tf.string),
                                               'seq_len': tf.FixedLenFeature([], tf.int64),
                                               'label_mor': tf.FixedLenFeature([], tf.int64),
                                               'label_dis': tf.FixedLenFeature([], tf.int64)
                                           })
        index = tf.reshape(tf.decode_raw(features['index'], tf.float32), [max_len, dim[0]])
        medicine = tf.reshape(tf.decode_raw(features['medicine'], tf.float32), [max_len, dim[1]])
        label_mor = tf.to_int32(features['label_mor'])
        label_dis = tf.to_int32(features['label_dis'])
        seq_len = tf.to_int32(features['seq_len'])
        patient_id = features['patient_id']
        return patient_id, index, medicine, seq_len, label_mor, label_dis

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


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle, logger):
    losses = []
    pre_mors, ref_mors = [], []
    pre_diss, ref_diss = [], []
    metrics_mor, metrics_dis = {}, {}
    # pre_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    # ref_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    for _ in range(num_batches):
        patient_ids, loss, mor_preds, dis_preds, seq_lens = sess.run(
            [model.id, model.loss, model.mor_preds, model.dis_preds, model.seq_len],
            feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        for pid, mor_pred, dis_pred, seq_len in zip(patient_ids, mor_preds, dis_preds, seq_lens):
            sample = eval_file[str(pid)]

            ref_mors += [sample['label_mor']] * seq_len
            pre_mors += mor_pred[:seq_len].tolist()
            pre_diss.append(dis_pred)
            ref_diss.append(sample['label_dis'])
            # for k, v in pre_points.items():
            #     if seq_len >= k:
            #         v.append(mor_pred[k - 1])
            #         ref_points[k].append(sample['label_mor'])

    batch_loss = np.mean(losses)
    metrics_mor['acc'] = accuracy_score(ref_mors, pre_mors)
    metrics_mor['roc'] = roc_auc_score(ref_mors, pre_mors)
    (precisions, recalls, thresholds) = precision_recall_curve(ref_mors, pre_mors)
    metrics_mor['prc'] = auc(recalls, precisions)
    metrics_mor['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    # for k, v in pre_points.items():
    #     logger.info('{} hour confusion matrix. AUCROC : {}'.format(int(k / 3), roc_auc_score(ref_points[k], v)))
    #     logger.info(confusion_matrix(ref_points[k], v))
    logger.info('Mortality confusion matrix')
    logger.info(confusion_matrix(ref_mors, pre_mors))
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=batch_loss), ])
    mor_acc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/mor/acc'.format(data_type), simple_value=metrics_mor['acc']), ])
    mor_auc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/mor/roc'.format(data_type), simple_value=metrics_mor['roc']), ])
    mor_prc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/mor/prc'.format(data_type), simple_value=metrics_mor['prc']), ])

    metrics_dis['acc'] = accuracy_score(ref_diss, pre_diss)

    metrics_dis['roc'] = roc_auc_score(ref_diss, pre_diss)
    (precisions, recalls, thresholds) = precision_recall_curve(ref_diss, pre_diss)
    metrics_dis['prc'] = auc(recalls, precisions)
    metrics_dis['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    logger.info('Disease confusion matrix')
    logger.info(confusion_matrix(ref_diss, pre_diss))
    dis_acc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/dis/acc'.format(data_type), simple_value=metrics_dis['acc']), ])
    dis_auc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/dis/roc'.format(data_type), simple_value=metrics_dis['roc']), ])
    dis_prc = tf.Summary(
        value=[tf.Summary.Value(tag='{}/dis/prc'.format(data_type), simple_value=metrics_dis['prc']), ])

    return batch_loss, (metrics_mor, metrics_dis), (loss_sum, mor_acc, mor_auc, mor_prc, dis_acc, dis_auc, dis_prc)
