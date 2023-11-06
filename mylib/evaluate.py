"""Evaluates the model."""


# %% Imports
# %%% Py3 Standard
import time

# %%% 3rd Party
import numpy as np
import torch
import torch.nn.functional as F

# %%% User Defined
from mylib import utils


# %% Functions
# def my_evaluate(model, test_loader, n_drugs, device="cpu"):
#     model.eval()
    # see orig code

    # return ja, prauc, avg_p, avg_r, avg_f1, avg_med, ddi_rate

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

    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(data_loader):

            if isinstance(input_, tuple):
                input_ = tuple([
                    e.to(device) if isinstance(e, torch.Tensor) else e
                    for e in input_
                ])
            else:
                input_ = input_.to(device)
            target = target.to(device)

            output = model(input_)

            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(
                utils.compute_batch_accuracy(output, target).item(),
                target.size(0),
            )

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print(
                    f'Test: [{i}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'
                )

    return losses.avg, accuracy.avg, results

# def orig_evaluate(model, test_loader, n_drugs, device="cpu"):
#     model.eval()
#     ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
#     smm_record = []
#     med_cnt, visit_cnt = 0, 0
#     for step, adm in enumerate(test_loader):
#         y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

#         syms: torch.Tensor = torch.tensor(adm[0]).to(device)
#         drugs: torch.Tensor = torch.tensor(adm[2]).to(device)
#         scores = model.evaluate(syms, device=device)
#         # scores = 2 * torch.softmax(scores, dim=-1) - 1

#         y_gt_tmp = np.zeros(n_drugs)
#         y_gt_tmp[drugs.cpu().numpy()] = 1
#         y_gt.append(y_gt_tmp)

#         result = torch.sigmoid(scores).detach().cpu().numpy()
#         y_pred_prob.append(result)
#         y_pred_tmp = result.copy()
#         y_pred_tmp[y_pred_tmp >= 0.5] = 1
#         y_pred_tmp[y_pred_tmp < 0.5] = 0
#         y_pred.append(y_pred_tmp)

#         y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
#         y_pred_label.append(sorted(y_pred_label_tmp))
#         visit_cnt += 1
#         med_cnt += len(y_pred_label_tmp)

#         smm_record.append(y_pred_label)
#         adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
#             np.array(y_gt),
#             np.array(y_pred),
#             np.array(y_pred_prob),
#         )

#         ja.append(adm_ja)
#         prauc.append(adm_prauc)
#         avg_p.append(adm_avg_p)
#         avg_r.append(adm_avg_r)
#         avg_f1.append(adm_avg_f1)
#     # print(y_pred_label)
#     ddi_rate = ddi_rate_score(
#         smm_record,
#         path='datasets/MIMIC3/ddi_A_final.pkl',
#     )
#     # ddi_rate = 0
#     return (
#         np.mean(ja),
#         np.mean(prauc),
#         np.mean(avg_p),
#         np.mean(avg_r),
#         np.mean(avg_f1),
#         1.0 * med_cnt / visit_cnt, ddi_rate,
#     )
