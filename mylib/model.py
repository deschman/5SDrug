import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: implement their model.  see original paper and code base for reference.

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = nn.Linear(178, 80)
        self.mid = nn.Linear(80, 16)
        self.output = nn.Linear(16, 5)

    def forward(self, x):
        x = self.hidden(x.to(torch.float))
        x = torch.sigmoid(x)
        x = self.mid(x.to(torch.float))
        x = torch.sigmoid(x)
        x = self.output(x.to(torch.float))
        #print('out', x.shape)
        return x