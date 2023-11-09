"""Contains utilities required for model training and evaluation."""


# %% Imports
# %%% Py3 Standard
from typing import List, Set

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
        similar_sets: List[int] = []
        for idx, sym_batch in enumerate(sym_train):
            similar_sets.append(idx)
            sym_set: List[int] = sym_batch[0].tolist()
            max_intersection: int = 0
            for i, comparison_set in enumerate(sym_train):
                comp_sym_set: List[int] = comparison_set[0].tolist()
                if len(sym_set) <= 2 or len(comp_sym_set) <= 2 or i == idx: continue
                similar_symptoms: List[int] = [
                    symptom for i, symptom in enumerate(comp_sym_set)
                    if sym_set[i] == symptom
                ]
                if len(similar_symptoms) > max_intersection:
                    max_intersection = len(similar_symptoms)
                    similar_sets[idx] = i

        return similar_sets
