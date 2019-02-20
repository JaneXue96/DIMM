import os
import argparse
import logging
import ujson as json
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch_model import TCN
from torch_utils import FocalLoss, train_one_epoch, evaluate_one_epoch
from torch_loader import run_prepare, MyDataset, PadCollate

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
    parser.add_argument('--seed', type=int, default=23333,
                        help='random seed (default: 23333)')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--disable_cuda', action='store_true',
                                help='Disable CUDA')
    train_settings.add_argument('--lr', type=float, default=5e-4,
                                help='learning rate')
    train_settings.add_argument('--clip', type=float, default=-1,
                                help='gradient clip, -1 means no clip (default: 0.35)')
    train_settings.add_argument('--weight_decay', type=float, default=0.0001,
                                help='weight decay')
    train_settings.add_argument('--dropout', type=float, default=0.3,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_train', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--batch_eval', type=int, default=64,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=30,
                                help='train epochs')
    train_settings.add_argument('--optim', default='Adam',
                                help='optimizer type')
    train_settings.add_argument('--patience', type=int, default=2,
                                help='num of epochs for train patients')
    # 4019-60 41401-40 25000-40 5849-30
    train_settings.add_argument('--period', type=int, default=30,
                                help='period to save batch loss')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--max_len', type=int, default=720,
                                help='max length of sequence')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--use_cudnn', type=bool, default=True,
                                help='whether to use cudnn rnn')
    model_settings.add_argument('--layer_num', type=int, default=2,
                                help='num of layers')
    model_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    model_settings.add_argument('--capacity', type=int, default=20000,
                                help='Batch size of data set shuffle')
    model_settings.add_argument('--is_map', type=bool, default=False,
                                help='whether to encoding input')
    model_settings.add_argument('--is_point', type=bool, default=False,
                                help='whether to predict point label')
    model_settings.add_argument('--is_fc', type=bool, default=False,
                                help='whether to use focal loss')
    model_settings.add_argument('--is_atten', type=bool, default=False,
                                help='whether to use self attention')
    model_settings.add_argument('--is_gated', type=bool, default=False,
                                help='whether to use gated conv')
    model_settings.add_argument('--n_head', type=int, default=2,
                                help='attention head size (default: 2)')
    model_settings.add_argument('--n_kernel', type=int, default=3,
                                help='kernel size (default: 3)')
    model_settings.add_argument('--n_level', type=int, default=4,
                                help='# of levels (default: 10)')
    model_settings.add_argument('--n_filter', type=int, default=256,
                                help='number of hidden units per layer (default: 256)')
    model_settings.add_argument('--n_class', type=int, default=2,
                                help='class size (default: 2)')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='5849',
                               help='the task name')
    path_settings.add_argument('--model', default='TCN',
                               help='the model name')
    path_settings.add_argument('--raw_dir', default='data/raw_data/',
                               help='the dir to store raw data')
    path_settings.add_argument('--preprocessed_dir', default='data/preprocessed_data/torch/',
                               help='the dir to store prepared data')
    path_settings.add_argument('--outputs_dir', default='outputs/single_task/',
                               help='the dir of outputs')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def train(args, file_paths):
    logger = logging.getLogger('Medical')
    logger.info('Loading data sets...')
    train_set = MyDataset(file_paths.train_file)
    test_set = MyDataset(file_paths.test_file)
    train_loader = DataLoader(train_set, batch_size=args.batch_train, shuffle=True, num_workers=8, collate_fn=PadCollate())
    test_loader = DataLoader(test_set, batch_size=args.batch_eval, num_workers=8, collate_fn=PadCollate())
    # logger.info('Loading meta...')
    # with open(file_paths.meta, 'rb') as fh:
    #     meta = pkl.load(fh)
    # fh.close()
    # logger.info('Loading shape meta...')
    # with open(file_paths.shape_meta, 'rb') as fh:
    #     shape_meta = pkl.load(fh)
    # fh.close()
    # dim = shape_meta['dim']
    # train_num = meta['train_total']
    # eval_num = meta['test_total']
    # logger.info('Num train data {} Num eval data {}'.format(train_num, eval_num))
    # logger.info('Index dim {} Medicine dim {}'.format(dim[0], dim[1]))

    logger.info('Initialize the model...')
    model = TCN(input_size=205 + 241, output_size=args.n_class, n_channel=[args.n_filter]*args.n_level,
                n_kernel=args.n_kernel, dropout=args.dropout, logger=logger).to(device=args.device)
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=args.patience, verbose=True)
    # torch.backends.cudnn.benchmark = True
    max_acc, max_roc, max_prc, max_pse, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
    FALSE = []
    for ep in range(1, args.epochs + 1):
        logger.info('Training the model for epoch {}'.format(ep))
        avg_loss = train_one_epoch(model, optimizer, train_loader, args, logger)
        logger.info('Epoch {} AvgLoss {}'.format(ep, avg_loss))

        logger.info('Evaluating the model for epoch {}'.format(ep))
        eval_metrics = evaluate_one_epoch(model, test_loader, args.device, 'eval', args.is_point, logger)
        logger.info('Dev Loss: {}'.format(eval_metrics['loss']))
        logger.info('Dev Acc: {}'.format(eval_metrics['acc']))
        logger.info('Dev AUROC: {}'.format(eval_metrics['roc']))
        logger.info('Dev AUPRC: {}'.format(eval_metrics['prc']))
        logger.info('Dev PSe: {}'.format(eval_metrics['pse']))
        FALSE.append({'Epoch': ep, 'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']})
        max_acc = max(eval_metrics['acc'], max_acc)
        max_roc = max(eval_metrics['roc'], max_roc)
        max_prc = max(eval_metrics['prc'], max_prc)
        max_pse = max(eval_metrics['pse'], max_pse)
        dev_sum = eval_metrics['roc'] + eval_metrics['prc'] + eval_metrics['pse']
        if dev_sum > max_sum:
            max_sum = dev_sum
            max_epoch = ep
        scheduler.step(metrics=eval_metrics['roc'])

    logger.info('Max AUROC {}'.format(max_roc))
    logger.info('Max AUPRC {}'.format(max_prc))
    logger.info('Max Acc {}'.format(max_acc))
    logger.info('Max PSE {}'.format(max_pse))
    logger.info('Max Epoch {}'.format(max_epoch))
    with open(os.path.join(args.result_dir, 'FALSE.json'), 'w') as f:
        for record in FALSE:
            f.write(json.dumps(record) + '\n')
    f.close()


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
    torch.manual_seed(args.seed)
    args.device = None
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    logger.info('Preparing the directories...')
    args.raw_dir = args.raw_dir + args.task
    args.preprocessed_dir = args.preprocessed_dir + args.task
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)
    for dir_path in [args.raw_dir, args.preprocessed_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_file = os.path.join(args.preprocessed_dir, 'train.npy')
            self.test_file = os.path.join(args.preprocessed_dir, 'test.npy')
            # 计数文件
            self.meta = os.path.join(args.preprocessed_dir, 'meta.pkl')
            self.shape_meta = os.path.join(args.preprocessed_dir, 'shape_meta.pkl')

    file_paths = FilePaths()
    if args.prepare:
        run_prepare(args)
    if args.train:
        train(args, file_paths)


if __name__ == '__main__':
    run()
