import numpy as np
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc


def get_batch(samples, dim, device):
    seq_lens = [sample['length'] for sample in samples]
    max_len = max(seq_lens)
    indexes, medicines, labels = [], [], []
    for sample in samples:
        index = np.zeros([max_len, dim[0]], dtype=np.float32)
        medicine = np.zeros([max_len, dim[1]], dtype=np.float32)
        seq_len = sample['length']
        index[:seq_len] = sample['index'][:seq_len]
        medicine[:seq_len] = sample['medicine'][:seq_len]

        indexes.append(index)
        medicines.append(medicine)
        labels.append([sample['label']] * max_len)
    indexes = np.asarray(indexes, dtype=np.float32)
    medicines = np.asarray(medicines, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    return torch.from_numpy(indexes).to(device), torch.from_numpy(medicines).to(device), \
           torch.from_numpy(labels).to(device), torch.from_numpy(seq_lens).to(device)


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))

    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=len(logits_flat.size()) - 1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()

    return loss


def evaluate_batch(model, data_num, batch_size, eval_file, dim, device, data_type, is_point, logger):
    losses = []
    pre_labels, pre_scores, ref = [], [], []
    fp = []
    fn = []
    metrics = {}
    pre_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    ref_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    model.eval()
    for batch_idx, batch in enumerate(range(0, data_num, batch_size)):
        start_idx = batch
        end_idx = start_idx + batch_size
        indexes, medicines, labels, seq_lens = get_batch(eval_file[start_idx:end_idx], dim, device)
        outputs = model(indexes, medicines)
        outputs = outputs.detach()
        loss = compute_loss(logits=outputs, target=labels, length=seq_lens).item()
        losses.append(loss)
        output_labels = torch.max(outputs.cpu(), 2)[1].numpy()
        output_scores = outputs.cpu()[:, :, 1].numpy()
        labels = labels.cpu().numpy()
        seq_lens = seq_lens.cpu().numpy()

        for pre_label, pre_score, label, seq_len in zip(output_labels, output_scores, labels, seq_lens):
            if is_point:
                pre_labels.append(pre_label[seq_len - 1])
                ref.append(label[seq_len - 1])
            else:
                pre_labels += pre_label[:seq_len].tolist()
                pre_scores += pre_score[:seq_len].tolist()
                ref += label[:seq_len].tolist()

            # if data_type == 'dev':
            #     if sample['label'] == 1 and final_pre_label == 0:
            #         fp.append(sample['name'])
            #     if sample['label'] == 0 and final_pre_label == 1:
            #         fn.append(sample['name'])
            for k, v in pre_points.items():
                if seq_len >= k:
                    v.append(pre_label[k - 1])
                    ref_points[k].append(label[k - 1])

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(ref, pre_labels)
    metrics['roc'] = roc_auc_score(ref, pre_scores)
    (precisions, recalls, thresholds) = precision_recall_curve(ref, pre_scores)
    metrics['prc'] = auc(recalls, precisions)
    metrics['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    if data_type == 'eval':
        metrics['fp'] = fp
        metrics['fn'] = fn
    for k, v in pre_points.items():
        logger.info('{} hour confusion matrix. AUCROC : {}'.format(int(k / 3), roc_auc_score(ref_points[k], v)))
        logger.info(confusion_matrix(ref_points[k], v))
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(ref, pre_labels))
    return metrics
    # tn, fp, fn, tp = confusion_matrix(auc_ref, auc_pre).ravel()
    # loss_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=metrics['loss']), ])
    # acc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=metrics['acc']), ])
    # auc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/roc'.format(data_type), simple_value=metrics['roc']), ])
    # prc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/prc'.format(data_type), simple_value=metrics['prc']), ])
    # return metrics, (loss_sum, acc_sum, auc_sum, prc_sum)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = functional.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
