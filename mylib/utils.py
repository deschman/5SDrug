"""Contains utilities required for model training and evaluation."""


# %% Imports
# %%% Py3 Standard
from pathlib import Path
from typing import List

# %%% 3rd Party
import dill
import torch
import torch.nn.functional as F
import numpy as np


# %% Classes
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val: float = val
        self.sum += val * n
        self.count += n
        self.avg: float = self.sum / self.count


# %% Functions
def compute_batch_accuracy(output: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        batch_size: int = target.size(0)
        correct: int = output.eq(target).sum(axis=0)

        return (correct * 100.0 / batch_size).mean()

def find_similar_sets(sym_train: torch.Tensor) -> List[int]:
    similar_sets: List[int] = []
    for idx, sym_batch in enumerate(sym_train):
        similar_sets.append(idx)
        sym_set: List[int] = sym_batch.tolist()
        max_intersection: int = 0

        for i, comparison_set in enumerate(sym_train):
            comp_sym_set: List[int] = comparison_set.tolist()
            if len(sym_set) <= 2 or len(comp_sym_set) <= 2 or i == idx:
                continue
            similar_symptoms: List[int] = [
                symptom for i, symptom in enumerate(comp_sym_set)
                if sym_set[i] == symptom
            ]
            if len(similar_symptoms) > max_intersection:
                max_intersection = len(similar_symptoms)
                similar_sets[idx] = i

    return similar_sets

def custom_criterion(
    scores: torch.Tensor,
    bpr: torch.Tensor,
    loss_ddi: float,
    drugs: torch.Tensor,
    alpha: float,
    beta: float,
    device: str,
):
    sig_scores: torch.Tensor = torch.sigmoid(scores)
    scores_sigmoid: torch.Tensor = torch.where(
        sig_scores == 0, torch.tensor(1.0).to(device),
        sig_scores,
    )

    bce_loss: torch.Tensor = F.binary_cross_entropy_with_logits(scores, drugs)
    entropy: torch.Tensor = -torch.mean(
        sig_scores * (torch.log(scores_sigmoid) - 1)
    )

    return bce_loss + 0.5 * entropy + alpha * bpr + beta * loss_ddi

def get_ddi_score(all_preds: List[List[int]]) -> float:
    data_path: Path = Path(__file__).parent.parent / 'data'
    ddi_set: np.ndarray = dill.load(
        open(
            data_path / 'ddi_A_final.pkl',
            'rb',
        )
    )

    all: int = 0
    interactions: int = 0
    for pred in all_preds:
        for i, med_i in enumerate(pred):
            for j, med_j in enumerate(pred):
                if j <= i:
                    continue
                all += 1
                if ddi_set[med_i, med_j] == 1 or ddi_set[med_j, med_i] == 1:
                    interactions += 1
    if all == 0:
        return 0
    return interactions / all
