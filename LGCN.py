import math
import torch
import torch.nn as nn

class LGCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(LGCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.p = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.p.size(1))
        self.p.data.uniform_(-stdv, stdv)
        if self.use_bias:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h_x, line_g):
        line = line_g @ h_x @ self.p  # 超边节点学习
        if self.use_bias:
            line += self.bias
        return line