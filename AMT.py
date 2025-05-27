import torch
import torch.nn as nn


class Transformer1(nn.Module):
    def __init__(self, embedding_dim=4900, hidden_dim=1024, num_heads=8, dropout=0.3):
        super(Transformer1, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.calculate_q1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_q1.weight, gain=1)
        self.calculate_q2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_q2.weight, gain=1)
        self.calculate_k1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_k1.weight, gain=1)
        self.calculate_k2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_k2.weight, gain=1)
        self.calculate_v1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_v1.weight, gain=1)
        self.calculate_v2 = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_v2.weight, gain=1)

        self.a_k = nn.Parameter(torch.rand(1))
        self.a_q = nn.Parameter(torch.rand(1))
        self.a_v = nn.Parameter(torch.rand(1))
        self.b_k = nn.Parameter(torch.rand(1))
        self.b_q = nn.Parameter(torch.rand(1))
        self.b_v = nn.Parameter(torch.rand(1))

        self.MultiheadAttention = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc.weight, nn.init.calculate_gain('leaky_relu'))
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight, nn.init.calculate_gain('leaky_relu'))
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight, nn.init.calculate_gain('leaky_relu'))

        self.dropout2 = nn.Dropout(p=dropout)

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.relu2 = nn.LeakyReLU()

    def forward(self, querys1, querys2, keys1, keys2, values1, values2, mask=None):
        paramsk = torch.stack([self.a_k, self.b_k])
        softmax_params = torch.softmax(paramsk, dim=0)
        self.a_k.data = softmax_params[0]
        self.b_k.data = softmax_params[1]

        paramsq = torch.stack([self.a_q, self.b_q])
        softmax_params = torch.softmax(paramsq, dim=0)
        self.a_q.data = softmax_params[0]
        self.b_q.data = softmax_params[1]

        paramsv = torch.stack([self.a_v, self.b_v])
        softmax_params = torch.softmax(paramsv, dim=0)
        self.a_v.data = softmax_params[0]
        self.b_v.data = softmax_params[1]

        query1 = self.a_q * self.calculate_q1(querys1)
        key1 = self.a_k * self.calculate_k1(keys1)
        value1 = self.a_v * self.calculate_v1(values1)

        query2 = self.b_q * self.calculate_q2(querys2)
        key2 = self.b_k * self.calculate_k2(keys2)
        value2 = self.b_v * self.calculate_v2(values2)

        query = query1 + query2
        key = key1 + key2
        value = value1 + value2

        output, _ = self.MultiheadAttention(query, key, value, key_padding_mask=mask)
        output = query + self.dropout(output)
        output = self.layer_norm(output)
        output = self.fc(output)
        tmp = output

        output = self.fc2(self.dropout2(self.relu2(self.fc1(output))))
        output = tmp + self.dropout2(output)
        output = self.layer_norm2(output)
        return output

class Transformer2(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, num_heads=8, dropout=0.3):
        super(Transformer2, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.calculate_q = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_q.weight, gain=1)

        self.calculate_k = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_k.weight, gain=1)

        self.calculate_v = nn.Linear(self.embedding_dim, self.hidden_dim)
        torch.nn.init.xavier_normal_(self.calculate_v.weight, gain=1)

        self.MultiheadAttention = nn.MultiheadAttention(self.hidden_dim, num_heads, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc.weight, nn.init.calculate_gain('leaky_relu'))
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight, nn.init.calculate_gain('leaky_relu'))

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight, nn.init.calculate_gain('leaky_relu'))

        self.dropout2 = nn.Dropout(p=dropout)

        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.relu2 = nn.LeakyReLU()

    def forward(self, query, key, value, mask=None):
        query = self.calculate_q(query)
        key = self.calculate_k(key)
        value = self.calculate_v(value)

        output, _ = self.MultiheadAttention(query, key, value, key_padding_mask=mask)
        output = query + self.dropout(output)
        output = self.layer_norm(output)
        output = self.fc(output)
        tmp = output

        output = self.fc2(self.dropout2(self.relu2(self.fc1(output))))
        output = tmp + self.dropout2(output)
        output = self.layer_norm2(output)
        return output