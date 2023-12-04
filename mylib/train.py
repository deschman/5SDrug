"""Trains the model."""


# %% Imports
# %%% Py3 Standard
import time
from typing import List, Tuple

# %%% 3rd Party
import numpy as np
import torch
from torch.utils.data import DataLoader

# %%% User Defined
from mylib import utils


# %% Functions
def hw4_train(
    model: torch.nn.Module,
    device: str,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    alpha: float,
    beta: float,
    print_freq: int = 10,
):
    batch_time: utils.AverageMeter = utils.AverageMeter()
    losses: utils.AverageMeter = utils.AverageMeter()
    accuracy: utils.AverageMeter = utils.AverageMeter()
    drug_count: utils.AverageMeter = utils.AverageMeter()

    model.train()

    end: float = time.time()
    for i, (input_, target) in enumerate(data_loader):
        input_: torch.Tensor = input_.to(device)
        similar_sets: List[int] = utils.find_similar_sets(input_)
        target: torch.Tensor = target.to(device)

        optimizer.zero_grad()
        output: Tuple[torch.Tensor, torch.Tensor, float] = model(input_, target, similar_sets)
        loss = utils.custom_criterion(
            output[0],
            output[1],
            output[2],
            target.float(),
            alpha,
            beta,
            device,
        )
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(
            utils.compute_batch_accuracy(output[0].round(), target).item(),
            target.size(0),
        )
        drug_count.update(output[0].round().max(axis=1)[0].mean().item())

        if i % print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                f'Drug Count {int(drug_count.val)} ({drug_count.avg:.3f})'
            )

    return losses.avg, accuracy.avg
