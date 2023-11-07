"""Trains the model."""


# %% Imports
# %%% Py3 Standard
import time

# %%% 3rd Party
import numpy as np
import torch

# %%% User Defined
from mylib import evaluate, utils


# %% Variables
ALPHA = 0.5
BETA = 1.0


# %% Functions
# def my_train(model, device, optimizer, sym_train, drug_train, similar_sets_idx, data_eval, n_drug):

#     model.train()
#     losses = 0.0

#     # training loop
#     for i, (syms, drugs, similar_idx) in enumerate(zip(sym_train, drug_train, similar_sets_idx)):
#         model.zero_grad()
#         optimizer.zero_grad()
#         scores, bpr, loss_ddi = model(syms, drugs, similar_idx, device)

#         loss = evaluate.custom_criterion(scores, bpr, loss_ddi, drugs, ALPHA, BETA, device)
#         losses += loss.item() / syms.shape[0]
#         loss.backward()
#         optimizer.step()

#         train_accuracy = evaluate.my_evaluate(
#             model, data_eval, n_drug, device
#         )
#         ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate = train_accuracy

#     return losses, ja

def hw4_train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if isinstance(e, torch.Tensor) else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input[0], target)  # TODO: get similar symptom sets to feed model
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(utils.compute_batch_accuracy(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})')

    return losses.avg, accuracy.avg
