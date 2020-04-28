import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 use_dropout,
                 use_reconstruction):
        super().__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = lambda x: x

        if use_reconstruction:
            self.l3 = nn.Linear(hidden_size, input_size)

        self.use_reconstruction = use_reconstruction

    def forward(self, x):
        h = self.dropout(F.relu(self.l1(x)))
        y = self.l2(h)

        x_hat = None
        if self.use_reconstruction:
            x_hat = F.sigmoid(self.l3(h))

        return y, x_hat


class DietNetworks(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 sample_size, embedding_size,
                 use_dropout,
                 use_reconstruction):
        super().__init__()

        self.aux_e1 = nn.Linear(sample_size, embedding_size)
        self.aux_e2 = nn.Linear(embedding_size, hidden_size)

        self.l2 = nn.Linear(hidden_size, output_size)

        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = lambda x: x

        if use_reconstruction:
            self.aux_r1 = nn.Linear(sample_size, embedding_size)
            self.aux_r2 = nn.Linear(embedding_size, hidden_size)

        self.use_reconstruction = use_reconstruction

    def forward(self, x):
        x_t = torch.t(x)
        We = self.aux_e2(self.dropout(F.relu(self.aux_e1(x_t))))
        h = torch.matmul(x, We)
        y = self.l2(h)

        x_hat = None
        if self.use_reconstruction:
            Wr = self.aux_r2(self.dropout(F.relu(self.aux_r1(x_t))))
            x_hat = F.sigmoid(torch.matmul(h, torch.t(Wr)))

        return y, x_hat


class ModifiedDietNetworks(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 sample_size, embedding_size,
                 use_dropout,
                 use_reconstruction):
        super().__init__()

        self.Ue = nn.Parameter(torch.rand(1, input_size))
        self.aux_e1 = nn.Linear(sample_size, embedding_size)
        self.aux_e2 = nn.Linear(embedding_size, hidden_size)

        self.l2 = nn.Linear(hidden_size, output_size)

        if use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = lambda x: x

        if use_reconstruction:
            self.Ur = nn.Parameter(torch.ones(1, input_size))
            self.aux_r1 = nn.Linear(sample_size, embedding_size)
            self.aux_r2 = nn.Linear(embedding_size, hidden_size)

        self.use_reconstruction = use_reconstruction

    def forward(self, x):
        x_t = torch.t(x)
        Ue = self.Ue.expand_as(x)
        We = self.aux_e2(self.dropout(F.relu(self.aux_e1(x_t))))
        h = torch.matmul(x * self.Ue, We)
        y = self.l2(h)

        x_hat = None
        if self.use_reconstruction:
            Ur = self.Ur.expand_as(x)
            Wr = self.aux_r2(self.dropout(F.relu(self.aux_r1(x_t))))
            x_hat = F.sigmoid(Ur * torch.matmul(h, torch.t(Wr)))

        return y, x_hat
