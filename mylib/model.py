"""Implements the 5SDrug model."""


# %% Imports
# %%% Py3 Standard
from typing import List

# %%% 3rd Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import sparse


# %% Classes
# TODO: implement their model.  see original paper and code base for reference.
class Model(nn.Module):
    def __init__(
        self,
        n_symptoms: int,
        n_drugs: int,
        ddi,
        symptom_set: List[sparse.coo_array],
        drug_set: List[sparse.coo_array],
        embed_dim: int,
    ):
        super(Model, self).__init__()
        self.input_embedding = nn.Embedding(n_symptoms, embed_dim)
        self.output_embedding = nn.Embedding(n_drugs, embed_dim)

    def forward(self, x):
        x = self.hidden(x.to(torch.float))
        x = F.sigmoid(x)
        x = self.mid(x.to(torch.float))
        x = F.sigmoid(x)
        x = self.output(x.to(torch.float))
        return x

# TODO: implement their attention aggregation layer
class Attention(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(embed_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

# TODO: implement their optimizer
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)
