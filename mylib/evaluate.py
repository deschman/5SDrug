"""Evaluates the model."""


# %% Imports
# %%% Py3 Standard
import time

# %%% 3rd Party
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

# %%% User Defined
from mylib import utils


# %% Functions
def custom_criterion(scores, bpr, loss_ddi, drugs, alpha, beta, device):
    sig_scores = torch.sigmoid(scores)
    scores_sigmoid = torch.where(sig_scores == 0, torch.tensor(1.0).to(device), sig_scores)

    bce_loss = F.binary_cross_entropy_with_logits(scores, drugs)
    entropy = -torch.mean(sig_scores * (torch.log(scores_sigmoid) - 1))
    loss = bce_loss + 0.5 * entropy + alpha * bpr + beta * loss_ddi
    return loss

def hw4_evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()

    results = []

    model.eval()

    f1_scores = []
    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(data_loader):

            input_ = input_.to(device)
            similar = utils.find_similar_sets(input_)
            target = target.to(device)

            output = model(input_, target, similar)

            loss = criterion(output[0], target.squeeze(1).float())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(
                utils.compute_batch_accuracy(output[0], target.squeeze(1)).item(),
                target.size(0),
            )

            y_true = target.detach().to('cpu').numpy().tolist()
            y_true = [y[0] for y in y_true]
            y_pred = output[0].detach().to('cpu').numpy().tolist()
            # reduce to optimistic prediction
            y_pred = [round(p[i]) for i, p in zip(np.argmax(y_pred, axis=1), y_pred)]
            results.extend(list(zip(y_true, y_pred)))
            f1_scores.append(f1_score(y_true, y_pred, average='macro'))
            print(sum(f1_scores) / len(f1_scores))

            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'
                )

    return losses.avg, accuracy.avg, results
