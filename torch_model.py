import torch
from torch import nn
from torch_tcn import TemporalConvNet
from torch_attention import Multihead_Attention, FeedForward
from time import time


class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_channel, n_kernel, dropout, logger):
        super(TCN, self).__init__()
        start_t = time()
        self.tcn = TemporalConvNet(input_size, n_channel, kernel_size=n_kernel, dropout=dropout)
        # self.mh_att = Multihead_Attention(input_size, num_heads=1, causality=True)
        # self.ffn = FeedForward(n_channel[-1], num_units=[2*n_channel[-1], n_channel[-1]])
        self.linear = nn.Linear(n_channel[-1], output_size)
        self.init_weights()
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, index, medicine):
        x = torch.cat([index, medicine], dim=2)
        # x = self.mh_att(x, x, x)
        y = self.tcn(x.transpose(2, 1))
        y = y.transpose(1, 2)
        # y = self.ffn(y)
        return self.linear(y)
