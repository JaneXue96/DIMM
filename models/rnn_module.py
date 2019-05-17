import tensorflow as tf
import tensorflow.contrib as tc
# from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
# from tensorflow.contrib import cudnn_rnn


def cu_rnn(rnn_type, inputs, hidden_size, batch_size, training, layer_num=1):
    if not rnn_type.startswith('bi'):
        cell = get_cu_cell(rnn_type, hidden_size, layer_num, 'unidirectional')
        inputs = tf.transpose(inputs, [1, 0, 2])
        c = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        h = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        outputs, state = cell(inputs)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        cell = get_cu_cell(rnn_type, hidden_size, layer_num, 'bidirectional')
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs, state = cell(inputs)
        # if concat:
        #     state = tf.concat([state_fw, state_bw], 1)
        # else:
        #     state = state_fw + state_bw
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, state


def get_cu_cell(rnn_type, hidden_size, layer_num=1, direction='bidirectional'):
    if rnn_type.endswith('lstm'):
        cudnn_cell = tc.cudnn_rnn.CudnnLSTM(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                            dropout=0)
    elif rnn_type.endswith('gru'):
        cudnn_cell = tc.cudnn_rnn.CudnnGRU(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                           dropout=0)
    elif rnn_type.endswith('rnn'):
        cudnn_cell = tc.cudnn_rnn.CudnnRNNTanh(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                               dropout=0)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    return cudnn_cell


def nor_rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    Args:
        rnn_type: the type of rnn
        inputs: padded inputs into rnn
        length: the valid length of the inputs
        hidden_size: the size of hidden units
        layer_num: multiple rnn layer are stacked if layer_num > 1
        dropout_keep_prob:
        concat: When the rnn is bidirectional, the forward outputs and backward outputs are
                concatenated if this is True, else we add them.
    Returns:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cells = tc.rnn.MultiRNNCell([get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)],
                                    state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cells, inputs, sequence_length=length, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        if layer_num > 1:
            cell_fw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            cell_bw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            outputs, state_fw, state_bw = tc.rnn.stack_bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
        else:
            cell_fw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            cell_bw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
        # state_fw, state_bw = state
        # if rnn_type.endswith('lstm'):
        #     c_fw, h_fw = state_fw
        #     c_bw, h_bw = state_bw
        #     state_fw, state_bw = h_fw, h_bw
        # if concat:
        #     outputs = tf.concat(outputs, 2)
        #     state = tf.concat([state_fw, state_bw], 1)
        # else:
        #
        #  = outputs[0] + outputs[1]
        #     state = state_fw + state_bw
    return outputs


def get_nor_cell(rnn_type, hidden_size, dropout_keep_prob=None):
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    elif rnn_type.endswith('sru'):
        cell = tc.rnn.SRUCell(num_units=hidden_size)
    elif rnn_type.endswith('indy'):
        cell = tc.rnn.IndyGRUCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    return cell
