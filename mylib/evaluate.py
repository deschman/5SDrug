"""Evaluates the model."""


# %% Imports
# %%% Py3 Standard
import time
from typing import List, Tuple

# %%% 3rd Party
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# %%% User Defined
from mylib import utils


# %% Functions
def hw4_evaluate(
    model: torch.nn.Module,
    device: str,
    data_loader: DataLoader,
    alpha: float,
    beta: float,
    print_freq: int = 10,
):
    batch_time: utils.AverageMeter = utils.AverageMeter()
    losses: utils.AverageMeter = utils.AverageMeter()
    accuracy: utils.AverageMeter = utils.AverageMeter()
    drug_count: utils.AverageMeter = utils.AverageMeter()

    results: List[Tuple[int, int]] = []
    all_preds: List[int] = []

    model.eval()

    f1_scores = []
    with torch.no_grad():
        end: float = time.time()
        for i, (input_, target) in enumerate(data_loader):

            input_: torch.Tensor = input_.to(device)
            similar: List[int] = utils.find_similar_sets(input_)
            target: torch.Tensor = target.to(device)

            output: Tuple[torch.Tensor, torch.Tensor, float] = model(
                input_,
                target,
                similar,
            )

            loss: float = utils.custom_criterion(
                output[0],
                output[1],
                output[2],
                target.float(),
                alpha,
                beta,
                device,
            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(
                utils.compute_batch_accuracy(output[0].round(), target).item(),
                target.size(0),
            )
            drug_count.update(output[0].round().max(axis=1)[0].mean().item())

            y_true: List[int] = target.detach().to('cpu').numpy().tolist()
            y_true = [y[0] for y in y_true]
            y_pred: List[int] = output[0].detach().to('cpu').numpy().tolist()
            # reduce to optimistic prediction
            y_pred = [round(p[i]) for i, p in zip(np.argmax(y_pred, axis=1), y_pred)]
            all_preds.append(y_pred)
            results.extend(list(zip(y_true, y_pred)))
            f1_scores.append(f1_score(y_true, y_pred, average='macro'))

            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                    f'Drug Count {int(drug_count.val)} ({drug_count.avg:.3f})\t'
                    f'F1 {sum(f1_scores) / len(f1_scores):.3f}'
                )

    ddi_score: float = utils.get_ddi_score(all_preds)

    return (
        sum(f1_scores) / len(f1_scores),
        losses.avg,
        accuracy.avg,
        drug_count.avg,
        ddi_score,
    )
