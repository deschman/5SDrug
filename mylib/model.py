"""Implements the 5SDrug model."""


# %% Imports
# %%% Py3 Standard
import math
from typing import List, Tuple, Dict, Any, Callable

# %%% 3rd Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from scipy import sparse


# %% Classes
class Model(nn.Module):
    def __init__(
        self,
        n_symptoms: int,
        n_drugs: int,
        ddi: np.ndarray,
        embed_dim: int,
    ):
        super(Model, self).__init__()
        self.n_sym: int = n_symptoms
        self.n_drug: int = n_drugs
        self.embed_dim: int = embed_dim
        self.sym_embeddings: nn.Embedding = nn.Embedding(
            self.n_sym,
            self.embed_dim,
        )
        self.drug_embeddings: nn.Embedding = nn.Embedding(
            self.n_drug,
            self.embed_dim,
        )
        self.sym_agg: Attention = Attention(self.embed_dim)
        self.tensor_ddi_adj: torch.Tensor = torch.tensor(ddi)
        self.sparse_ddi_adj: sparse.csr_matrix = sparse.csr_matrix(ddi)

        self.init_parameters()

    def init_parameters(self):
        stdv: float = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        syms: torch.Tensor,
        drugs: torch.Tensor,
        similar_idx: List[int],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        '''
        :param syms: [batch_size, sym_set_size]
        :param drugs: [batch_size, num_drugs]
        :param similar_idx: [batch_size]
        :param device: 'cpu' or 'gpu
        :return:
        '''
        all_drugs: torch.Tensor = torch.tensor(range(self.n_drug)).to(device)
        sym_embeds: torch.Tensor = self.sym_embeddings(syms.long())
        all_drug_embeds: torch.Tensor = self.drug_embeddings(all_drugs)
        s_set_embeds: torch.Tensor = self.sym_agg(sym_embeds)
        # s_set_embeds = torch.mean(sym_embeds, dim=1)
        all_drug_embeds: torch.Tensor = all_drug_embeds.repeat(
            s_set_embeds.shape[0],
            1,
            1,
        )

        scores: torch.Tensor = torch.bmm(
            s_set_embeds.unsqueeze(1),
            all_drug_embeds.transpose(-1, -2),
        ).squeeze(-2)  # [batch_size, n_drug]
        scores_aug: float = 0.0
        batch_neg: float = 0.0

        neg_pred_prob: torch.Tensor = torch.sigmoid(scores)
        neg_pred_prob = torch.mm(neg_pred_prob.transpose(-1, -2), neg_pred_prob)  # (voc_size, voc_size)
        batch_neg = 0.00001 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        if syms.shape[0] > 2 and syms.shape[1] > 2:
            scores_aug: torch.Tensor = self.intraset_augmentation(
                syms,
                drugs,
                all_drug_embeds,
                similar_idx,
                device,
            )
            batch_neg += self.intersect_ddi(
                syms,
                drugs,
                all_drug_embeds,
                similar_idx,
                device,
            )

        return scores, scores_aug, batch_neg

    def evaluate(self, syms: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        sym_embeds: torch.Tensor = self.sym_embeddings(syms.long())
        drug_embeds: torch.Tensor = self.drug_embeddings(
            torch.arange(
                0,
                self.n_drug,
            ).long().to(device)
        )
        s_set_embed: torch.Tensor = self.sym_agg(sym_embeds).unsqueeze(0)
        scores: torch.Tensor = torch.mm(
            s_set_embed,
            drug_embeds.transpose(-1, -2),
        ).squeeze(0)

        return scores

    def intraset_augmentation(
        self,
        syms: torch.Tensor,
        drugs: torch.Tensor,
        all_drug_embeds: torch.Tensor,
        similar_idx: List[int],
        device: str = 'cpu',
    ) -> torch.Tensor:
        selected_drugs: torch.Tensor = drugs[similar_idx]
        r: torch.Tensor = torch.tensor(
            range(drugs.shape[0])
        ).to(device).unsqueeze(1)
        sym_multihot: torch.Tensor = torch.zeros((
            drugs.shape[0],
            self.n_sym,
        )).to(device)
        selected_sym_multihot: torch.Tensor = torch.zeros((
            drugs.shape[0],
            self.n_sym,
        )).to(device)
        sym_multihot[r, syms] = 1
        selected_sym_multihot[r, syms[similar_idx]] =1

        common_sym: torch.Tensor = sym_multihot * selected_sym_multihot
        common_sym_sq: torch.Tensor = common_sym.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        all_sym_embeds: torch.Tensor = self.sym_embeddings(
            torch.tensor(
                range(
                    self.n_sym
                )
            ).to(device)
        ).unsqueeze(0).expand_as(common_sym_sq)
        common_sym_embeds: torch.Tensor = common_sym_sq * all_sym_embeds
        common_set_embeds: torch.Tensor = self.sym_agg(common_sym_embeds, common_sym)
        common_drug: torch.Tensor = drugs * selected_drugs
        diff_drug: torch.Tensor = drugs - selected_drugs
        diff_drug[diff_drug == -1] = 1

        common_drug_sum: torch.Tensor = torch.sum(common_drug, -1, True)
        diff_drug = torch.sum(diff_drug, -1, True)
        common_drug_sum[common_drug_sum == 0] = 1
        diff_drug[diff_drug == 0] = 1

        scores: torch.Tensor = torch.bmm(
            common_set_embeds.unsqueeze(1),
            all_drug_embeds.transpose(-1, -2),
        ).squeeze(1)

        return F.binary_cross_entropy_with_logits(
            scores,
            common_drug.squeeze(1).float()
        )

    def intersect_ddi(
        self,
        syms: List[int],
        drugs: torch.Tensor,
        all_drug_embeds: torch.Tensor,
        similar_idx: List[int],
        device: str = 'cpu',
    ) -> torch.Tensor:
        selected_drugs: torch.Tensor = drugs[similar_idx]
        r: torch.Tensor = torch.tensor(
            range(drugs.shape[0])
        ).to(device).unsqueeze(1)
        sym_multihot: torch.Tensor = torch.zeros((
            drugs.shape[0],
            self.n_sym,
        )).to(device)
        selected_sym_multihot: torch.Tensor = torch.zeros((
            drugs.shape[0],
            self.n_sym,
        )).to(device)
        sym_multihot[r, syms] = 1
        selected_sym_multihot[r, syms[similar_idx]] = 1

        common_sym: torch.Tensor = sym_multihot * selected_sym_multihot
        common_sym_sq: torch.Tensor = common_sym.unsqueeze(-1).repeat(
            1,
            1,
            self.embed_dim,
        )
        all_sym_embeds: torch.Tensor = self.sym_embeddings(
            torch.tensor(
                range(self.n_sym)
            ).to(device)
        ).unsqueeze(0).expand_as(common_sym_sq)
        common_sym_embeds: torch.Tensor = common_sym_sq * all_sym_embeds
        common_set_embeds = self.sym_agg(common_sym_embeds, common_sym)
        diff_drug: torch.Tensor = drugs - selected_drugs
        diff_drug_2: torch.Tensor = torch.zeros_like(diff_drug)
        diff_drug_2[diff_drug == -1], diff_drug[diff_drug == -1] = 1, 0

        diff_drug_exp: torch.Tensor  = diff_drug.unsqueeze(1)
        diff2_exp: torch.Tensor = diff_drug_2.unsqueeze(1)
        diff_drug = torch.sum(diff_drug, -1, True)
        diff_drug_2 = torch.sum(diff_drug_2, -1, True)
        diff_drug[diff_drug == 0] = 1
        diff_drug_2[diff_drug_2 == 0] = 1
        diff_drug_embed: torch.Tensor = torch.bmm(
            diff_drug_exp.float(),
            all_drug_embeds,
        ).squeeze() / diff_drug
        diff2_embed: torch.Tensor = torch.bmm(
            diff2_exp.float(),
            all_drug_embeds,
        ).squeeze() / diff_drug_2

        diff_score: torch.Tensor = torch.sigmoid(
            common_set_embeds * diff_drug_embed.float()
        )
        diff2_score: torch.Tensor = torch.sigmoid(
            common_set_embeds * diff2_embed.float()
        )

        return 0.0001 * torch.sum(diff2_score * diff_score)

class Attention(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int = 1):
        super(Attention, self).__init__()
        self.aggregation: nn.Linear = nn.Linear(embed_dim, output_dim)

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        weight: torch.Tensor = self.aggregation(x)  # [b, num_learn, 1]
        return torch.tanh(weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        device: str = 'cpu',
    ):
        x = x.squeeze(1)

        weight: torch.Tensor = None
        if mask is None:
            weight = torch.softmax(self._aggregate(x), dim=-2)
        else:
            mask = torch.where(
                mask == 0,
                torch.tensor(-1e7).to(device),
                torch.tensor(0.0).to(device),
            )
            weight = torch.softmax(
                self._aggregate(x).squeeze(-1) + mask,
                dim=-1
            ).float().unsqueeze(-1)
            weight = torch.where(
                torch.isnan(weight),
                torch.tensor(0.0).to(device),
                weight,
            )
        return torch.matmul(
            x.transpose(-1, -2).float(),
            weight,
        ).squeeze(-1)

class RAdam(Optimizer):
    def __init__(
        self,
        params: torch.ParameterDict,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: int = 0,
        degenerated_to_sgd: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        self.degenerated_to_sgd: bool = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if (
                    'betas' in param
                    and (
                        param['betas'][0] != betas[0]
                        or param['betas'][1] != betas[1]
                    )
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(RAdam, self).__setstate__(state)

    def step(self, closure: Callable = None) -> float:

        loss: float = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad: torch.Tensor = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32: torch.Tensor = p.data.float()

                state: Dict[str, Any] = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32
                    )

                exp_avg: float = state['exp_avg']
                exp_avg_sq: float = state['exp_avg_sq']
                betas: Tuple[float, float] = group['betas']
                beta1: float = betas[0]
                beta2: float = betas[1]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                step_size: float = -1.0
                if state['step'] == buffered[0]:
                    N_sma: float = buffered[1]
                    step_size = buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t: float = beta2 ** state['step']
                    N_sma_max: float = 2 / (1 - beta2) - 1
                    N_sma: float = (
                        N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    )
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) *
                            (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'],
                            p_data_fp32,
                        )
                    denom: float = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(
                        -step_size * group['lr'],
                        exp_avg,
                        denom,
                    )
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'],
                            p_data_fp32,
                        )
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
