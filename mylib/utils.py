"""Contains utilities required for model training and evaluation."""


# %% Imports
# %%% Py3 Standard
from typing import List

# %%% 3rd Party
import torch


# %% Classes
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# %% Functions
def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size

def find_similar_sets(sym_train: torch.Tensor):  # TODO: refactor this for our data shape
        similar_sets: List[List[int]] = [[] for _ in range(len(sym_train))]
        for i in range(len(sym_train)):
            for j in range(len(sym_train[i])):
                similar_sets[i].append(j)

        for idx, sym_batch in enumerate(sym_train):
            if len(sym_batch) <= 2 or len(sym_batch[0]) <= 2: continue
            batch_sets = [set(sym_set) for sym_set in sym_batch]
            for i in range(len(batch_sets)):
                max_intersection = 0
                for j in range(len(batch_sets)):
                    if i == j: continue
                    if len(batch_sets[i] & batch_sets[j]) > max_intersection:
                        max_intersection = len(batch_sets[i] & batch_sets[j])
                        similar_sets[idx][i] = j

        return similar_sets
