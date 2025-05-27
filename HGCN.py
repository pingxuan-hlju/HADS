import math
import torch
import torch.nn as nn


class HGCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(HGCN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.diag(torch.ones(4900)))  # 对角矩阵
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

    def forward(self, g1: torch.Tensor, g2: torch.Tensor, x: torch.Tensor):
        g = g1 @ self.weight @ g2 @ x @ self.p
        if self.use_bias:
            g += self.bias
        return g