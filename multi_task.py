import os
import argparse
import logging
import ujson as json
import numpy as np
import tensorflow as tf
from multi_preprocess import run_prepare
from models.bi_RNN import bi_RNN_Model
from models.sep_RNN import sep_RNN_Model
from models.TCN import TCN
from models.SAnD import SAND
from models.DIMM import DIMM_Model
from multi_util import get_record_parser, multi_evaluate, get_batch_dataset, get_dataset, evaluate_batch
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Medical')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    # 31113 6921 [14400, 120, 480, 108, 9] [29100, 300, 970, 216, 9]
    train_settings.add_argument('--num_steps', type=int, default=9600,
                                help='num of step')
    train_settings.add_argument('--period', type=int, default=80,
                                help='period to save batch loss')
    train_settings.add_argument('--checkpoint', type=int, default=320,
                                help='checkpoint for evaluation')
    train_settings.add_argument('--eval_num_batches', type=int, default=78,
                                help='num of batches for evaluation')

    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--lr', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0.0002,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.65,
                                help='dropout keep rate')
    train_settings.add_argument('--train_batch', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--dev_batch', type=int, default=15,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=30,
                                help='train epochs')
    train_settings.add_argument('--patience', type=int, default=3,
                                help='num of epochs for train patients')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--n_class', type=int, default=2,
                                help='class size (default: 2)')
    model_settings.add_argument('--max_len', type=int, default=720,
                                help='max length of sequence')
    model_settings.add_argument('--n_hidden', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--use_cudnn', type=bool, default=True,
                                help='whether to use cudnn rnn')
    model_settings.add_argument('--n_layer', type=int, default=2,
                                help='num of layers')
    model_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    model_settings.add_argument('--capacity', type=int, default=20000,
                                help='Batch size of data set shuffle')
    model_settings.add_argument('--is_map', type=bool, default=True,
                                help='whether to encoding input')
    model_settings.add_argument('--is_bi', type=bool, default=True,
                                help='whether to use bi-rnn')
    model_settings.add_argument('--is_point', type=bool, default=False,
                                help='whether to predict point label')
    model_settings.add_argument('--is_fc', type=bool, default=False,
                                help='whether to use focal loss')
    model_settings.add_argument('--ipt_att', type=bool, default=True,
                                help='whether to use input self attention')
    model_settings.add_argument('--block_ipt', type=int, default=4,
                                help='num of block for input attention')
    model_settings.add_argument('--head_ipt', type=int, default=1,
                                help='num of input attention head')
    model_settings.add_argument('--step_att', type=bool, default=True,
                                help='whether to use input step attention')
    model_settings.add_argument('--block_stp', type=int, default=2,
                                help='num of block for step attention')
    model_settings.add_argument('--head_stp', type=int, default=4,
                                help='num of step attention head')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='multi_task',
                               help='the task name')
    path_settings.add_argument('--model', default='DIMM',
                               help='the model name')
    path_settings.add_argument('--raw_dir', default='data/raw_data/',
                               help='the dir to store raw data')
    path_settings.add_argument('--preprocessed_dir', default='data/preprocessed_data/',
                               help='the dir to store prepared data')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def train(args, file_paths, dim):
    logger = logging.getLogger('Medical')
    logger.info('Loading train eval file...')
    with open(file_paths.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    logger.info('Loading dev eval file...')
    with open(file_paths.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    logger.info('Loading train meta...')
    with open(file_paths.train_meta, "r") as fh:
        train_meta = json.load(fh)
    logger.info('Loading dev meta...')
    with open(file_paths.dev_meta, "r") as fh:
        dev_meta = json.load(fh)
    train_total = train_meta['total']
    logger.info('Total train data {}'.format(train_total))
    dev_total = dev_meta['total']
    logger.info('Total dev data {}'.format(dev_total))
    logger.info('Index dim {} Medicine dim {}'.format(dim[0], dim[1]))
    tasks = ['5849', '25000']
    max_metrics = {}
    for t in tasks:
        max_metrics[t] = {'max_acc': 0.0, 'max_roc': 0.0, 'max_prc': 0.0, 'max_pse': 0.0, 'max_sum': 0.0, 'max_epoch': 0}

    parser = get_record_parser(args.max_len, dim)
    train_dataset = get_batch_dataset(file_paths.train_record_file, parser, args)
    dev_dataset = get_dataset(file_paths.dev_record_file, parser, args)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    logger.info('Initialize the model...')
    model = DIMM_Model(args, iterator, dim, logger)
    # model = sep_RNN_Model(args, iterator, dim, logger)
    # model = TCN(args, iterator, dim, logger)
    # model = SAND(args, iterator, dim, logger)
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=8,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # writer = tf.summary.FileWriter(args.summary_dir)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        max_sum, max_epoch, task_sum = 0, 0, 0
        train_roc = 0
        roc_save, patience = 0, 0
        # TODO
        max_acc, max_roc, max_prc, max_pse, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
        lr = args.lr
        if args.is_map:
            index_W = tf.get_default_graph().get_tensor_by_name('input_encoding/index/dense/W:0')
            medicine_W = tf.get_default_graph().get_tensor_by_name('input_encoding/medicine/dense/W:0')
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.n_batch, tf.constant(args.train_batch, dtype=tf.int32)))

        for global_step in range(1, args.num_steps + 1):
            sess.run(tf.assign(model.global_step, tf.constant(global_step + 1, dtype=tf.int32)))
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            if global_step % args.period == 0:
                logger.info('Period point {} Loss {}'.format(global_step, loss))
                # loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=loss), ])
                # writer.add_summary(loss_sum, global_step)

            if global_step % args.checkpoint == 0:
                logger.info('Evaluating the model for epoch {}'.format(global_step // args.checkpoint))
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                train_metrics = evaluate_batch(model, args.eval_num_batches, train_eval_file, sess, 'train',
                                               handle, train_handle, args.is_point, logger)
                logger.info('Train Loss - {}'.format(train_metrics['loss']))
                logger.info('Train Acc - {}'.format(train_metrics['acc']))
                logger.info('Train AUROC - {}'.format(train_metrics['roc']))
                logger.info('Train AUPRC - {}'.format(train_metrics['prc']))
                logger.info('Train PSe - {}'.format(train_metrics['pse']))
                # for s in summ:
                #     writer.add_summary(s, global_step)
                if train_metrics['roc'] > train_roc:
                    train_roc = train_metrics['roc']

                sess.run(tf.assign(model.n_batch, tf.constant(args.dev_batch, dtype=tf.int32)))
                dev_metrics = multi_evaluate(model, dev_total // args.dev_batch, dev_eval_file, sess,
                                             handle, dev_handle, args.is_point)
                # dev_metrics = evaluate_batch(model, dev_total // args.dev_batch, dev_eval_file, sess, 'dev',
                #                              handle, dev_handle, args.is_point, logger)
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
                roc = 0
                # logger.info('Dev Loss - {}'.format(dev_metrics['loss']))
                # logger.info('Dev Acc - {}'.format(dev_metrics['acc']))
                # logger.info('Dev AUROC - {}'.format(dev_metrics['roc']))
                # logger.info('Dev AUPRC - {}'.format(dev_metrics['prc']))
                # logger.info('Dev PSe - {}'.format(dev_metrics['pse']))
                # roc = dev_metrics['roc']
                for t in tasks:
                    logger.info('Task - {}'.format(t))
                    logger.info('Dev Acc - {}'.format(dev_metrics[t]['acc']))
                    logger.info('Dev AUROC - {}'.format(dev_metrics[t]['roc']))
                    logger.info('Dev AUPRC - {}'.format(dev_metrics[t]['prc']))
                    logger.info('Dev PSe - {}'.format(dev_metrics[t]['pse']))
                    roc += dev_metrics[t]['roc']
                    max_metrics[t]['max_acc'] = max((dev_metrics[t]['acc'], max_metrics[t]['max_acc']))
                    max_metrics[t]['max_roc'] = max(dev_metrics[t]['roc'], max_metrics[t]['max_roc'])
                    max_metrics[t]['max_prc'] = max(dev_metrics[t]['prc'], max_metrics[t]['max_prc'])
                    max_metrics[t]['max_pse'] = max(dev_metrics[t]['pse'], max_metrics[t]['max_pse'])
                    dev_sum = dev_metrics[t]['roc'] + dev_metrics[t]['prc'] + dev_metrics[t]['pse']
                    task_sum += dev_sum
                    if dev_sum > max_metrics[t]['max_sum']:
                        max_metrics[t]['max_sum'] = dev_sum
                        max_metrics[t]['max_epoch'] = global_step // args.checkpoint
                        # filename = os.path.join(args.model_dir, "model_{}.ckpt".format(global_step))
                        # saver.save(sess, filename)
                if roc > roc_save:
                    roc_save = roc
                    patience = 0
                else:
                    patience += 1
                if patience >= args.patience:
                    lr /= 2.0
                    roc_save = roc
                    patience = 0
                    logger.info('Learning rate reduced to {}'.format(lr))
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

                # max_acc = max((dev_metrics['acc'], max_acc))
                # max_roc = max(dev_metrics['roc'], max_roc)
                # max_prc = max(dev_metrics['prc'], max_prc)
                # max_pse = max(dev_metrics['pse'], max_pse)
                # dev_sum = dev_metrics['roc'] + dev_metrics['prc'] + dev_metrics['pse']
                # if dev_sum > max_sum:
                #     max_sum = dev_sum
                #     max_epoch = global_step // args.checkpoint
                #     if args.is_map:
                #         iw = sess.run(index_W)
                #         mw = sess.run(medicine_W)
                if task_sum > max_sum:
                    max_sum = task_sum
                    max_epoch = global_step // args.checkpoint
                    # filename = os.path.join(args.model_dir, "model_{}.ckpt".format(global_step))
                    # saver.save(sess, filename)
                    if args.is_map:
                        iw = sess.run(index_W)
                        mw = sess.run(medicine_W)
        logger.info('Max Train AUROC - {}'.format(train_roc))
        logger.info('Max Dev epoch - {}'.format(max_epoch))
        # logger.info('Max Acc - {}'.format(max_acc))
        # logger.info('Max AUROC - {}'.format(max_roc))
        # logger.info('Max AUPRC - {}'.format(max_prc))
        # logger.info('Max PSE - {}'.format(max_pse))
        for t in tasks:
            logger.info('Task - {}'.format(t))
            logger.info('Max Acc - {}'.format(max_metrics[t]['max_acc']))
            logger.info('Max AUROC - {}'.format(max_metrics[t]['max_roc']))
            logger.info('Max AUPRC - {}'.format(max_metrics[t]['max_prc']))
            logger.info('Max PSE - {}'.format(max_metrics[t]['max_pse']))
            logger.info('Max Epoch - {}'.format(max_metrics[t]['max_epoch']))
        if args.is_map:
            np.savetxt(os.path.join(args.result_dir, args.task + '_index_W.txt'), iw, fmt='%.6f', delimiter=',')
            np.savetxt(os.path.join(args.result_dir, args.task + '_medicine_W.txt'), mw, fmt='%.6f', delimiter=',')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger('Medical')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 是否存储日志
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('Preparing the directories...')
    args.preprocessed_dir = os.path.join(args.task, args.preprocessed_dir)
    args.model_dir = os.path.join(args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.task, args.model, args.result_dir)
    args.summary_dir = os.path.join(args.task, args.model, args.summary_dir)
    for dir_path in [args.raw_dir, args.preprocessed_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_record_file = os.path.join(args.preprocessed_dir, 'train.tfrecords')
            self.dev_record_file = os.path.join(args.preprocessed_dir, 'dev.tfrecords')
            self.test_record_file = os.path.join(args.preprocessed_dir, 'test.tfrecords')
            # 评估文件
            self.train_eval_file = os.path.join(args.preprocessed_dir, 'train_eval.json')
            self.dev_eval_file = os.path.join(args.preprocessed_dir, 'dev_eval.json')
            self.test_eval_file = os.path.join(args.preprocessed_dir, 'test_eval.json')
            # 计数文件
            self.train_meta = os.path.join(args.preprocessed_dir, 'train_meta.json')
            self.dev_meta = os.path.join(args.preprocessed_dir, 'dev_meta.json')
            self.test_meta = os.path.join(args.preprocessed_dir, 'test_meta.json')
            self.shape_meta = os.path.join(args.preprocessed_dir, 'shape_meta.json')

    file_paths = FilePaths()
    if args.prepare:
        max_seq_len, index_dim = run_prepare(args, file_paths)
        with open(file_paths.shape_meta, 'w') as fh:
            json.dump({'max_len': max_seq_len, 'dim': index_dim}, fh)
        fh.close()
    if args.train:
        with open(file_paths.shape_meta, 'r') as fh:
            shape_meta = json.load(fh)
        fh.close()
        train(args, file_paths, shape_meta['dim'])


if __name__ == '__main__':
    run()
