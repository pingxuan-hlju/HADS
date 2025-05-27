import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Gate, self).__init__()
        self.W1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_normal_(self.W1.weight, gain=1)
        self.W2 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_normal_(self.W2.weight, gain=1)
        self.W3 = nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.xavier_normal_(self.W3.weight, gain=1)
        self.W4 = nn.Linear(input_dim, output_dim, bias=False)
        torch.nn.init.xavier_normal_(self.W4.weight, gain=1)

    def forward(self, x1, x2):

        s = F.sigmoid(self.W1(x1) + self.W2(x2))
        fusion = F.tanh(self.W3(x1) + self.W4(x2))

        return  s * fusion