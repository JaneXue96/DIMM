import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix, precision_recall_curve, \
    auc


def get_record_parser(max_len, dim):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'patient_id': tf.FixedLenFeature([], tf.int64),
                                               'index': tf.FixedLenFeature([], tf.string),
                                               'medicine': tf.FixedLenFeature([], tf.string),
                                               'seq_len': tf.FixedLenFeature([], tf.int64),
                                               # 'score': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64)
                                           })
        index = tf.reshape(tf.decode_raw(features['index'], tf.float32), [max_len, dim[0]])
        medicine = tf.reshape(tf.decode_raw(features['medicine'], tf.float32), [max_len, dim[1]])
        # score = tf.reshape(tf.decode_raw(features['score'], tf.float32), [max_len])
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


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle, is_point, logger,
                   is_single=True):
    losses = []
    pre_scores, pre_labels, ref_labels = [], [], []
    fp = []
    fn = []
    names = []
    metrics = {}
    hour_metrics = []
    pre_points = {3 * k: [] for k in range(1, 73)}
    score_points = {3 * k: [] for k in range(1, 73)}
    ref_points = {3 * k: [] for k in range(1, 73)}
    for _ in range(num_batches):
        patient_ids, loss, labels, scores, seq_lens = sess.run([model.id, model.loss, model.pre_labels,
                                                                model.pre_scores, model.seq_len],
                                                               feed_dict={
                                                                   handle: str_handle} if handle is not None else None)
        losses.append(loss)
        for pid, pre_label, pre_score, seq_len in zip(patient_ids, labels, scores, seq_lens):
            sample = eval_file[str(pid)]
            if is_point:
                ref_labels.append(sample['label'])
                pre_labels.append(pre_label)
                pre_scores.append(pre_score)
                final_pre_label = pre_label
            else:
                ref_labels += [sample['label']] * seq_len
                pre_labels += pre_label[:seq_len].tolist()
                pre_scores += pre_score[:seq_len].tolist()
                final_pre_label = pre_label[seq_len - 1]
            if data_type == 'dev':
                if sample['label'] == 1 and final_pre_label == 0:
                    fp.append(sample['name'])
                if sample['label'] == 0 and final_pre_label == 1:
                    fn.append(sample['name'])
                for k, v in pre_points.items():
                    if seq_len >= k:
                        v.append(pre_label[k - 1])
                        score_points[k].append(pre_score[k - 1])
                        ref_points[k].append(sample['label'])
            else:
                names.append(sample['name'])
            # ref_score = sample['score']
            # mses.append(mean_squared_error(sample['score'][:seq_len], pre_score[:seq_len]))

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(ref_labels, pre_labels)
    metrics['roc'] = roc_auc_score(ref_labels, pre_scores)
    (precisions, recalls, thresholds) = precision_recall_curve(ref_labels, pre_scores)
    metrics['prc'] = auc(recalls, precisions)
    metrics['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    if data_type == 'dev':
        metrics['fp'] = fp
        metrics['fn'] = fn
        for k, v in pre_points.items():
            # logger.info('{} hour confusion matrix. AUCROC : {}'.format(int(k / 3), roc_auc_score(ref_points[k], v)))
            hour_metrics.append(cal_metrics(ref_points[k], score_points[k], v))
    else:
        metrics['name'] = names
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(ref_labels, pre_labels))
    # tn, fp, fn, tp = confusion_matrix(auc_ref, auc_pre).ravel()

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=metrics['loss']), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=metrics['acc']), ])
    # mse_sum = tf.Summary(value=[tf.Summary.Value(tag="eval/mse", simple_value=avg_mse), ])
    auc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/roc'.format(data_type), simple_value=metrics['roc']), ])
    prc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/prc'.format(data_type), simple_value=metrics['prc']), ])
    return metrics, hour_metrics, (loss_sum, acc_sum, auc_sum, prc_sum)


def cal_metrics(ref, pred_scores, pred_labels):
    metrics = {}
    metrics['acc'] = accuracy_score(ref, pred_labels)
    metrics['roc'] = roc_auc_score(ref, pred_scores)
    (precisions, recalls, thresholds) = precision_recall_curve(ref, pred_scores)
    metrics['prc'] = auc(recalls, precisions)
    metrics['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    return metrics
