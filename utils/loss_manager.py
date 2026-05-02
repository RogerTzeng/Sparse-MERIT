import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch.autograd as autograd
from collections import defaultdict

class LogManager:
    def __init__(self):
        self.log_book=defaultdict(lambda: [])
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            print(stat_type,":",stat, end=' / ')
        print(" ")

    def get_stat_str(self):
        result_str = ""
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            result_str += str(stat) + " / "
        return result_str

def CCC_loss(pred, lab, m_lab=None, v_lab=None, is_numpy=False):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    if is_numpy:
        pred = torch.Tensor(pred).float().cuda()
        lab = torch.Tensor(lab).float().cuda()
    
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc

def MSE_emotion(pred, lab):
    aro_loss = F.mse_loss(pred[:][0], lab[:][0])
    dom_loss = F.mse_loss(pred[:][1], lab[:][1])
    val_loss = F.mse_loss(pred[:][2], lab[:][2])

    return [aro_loss, dom_loss, val_loss]


def CE_weight_category(pred, lab, weights):
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    return criterion(pred, lab)


def calc_err(pred, lab):
    p = pred.detach()
    t = lab.detach()
    total_num = p.size()[0]
    ans = torch.argmax(p, dim=1)
    corr = torch.sum((ans==t).long())

    err = (total_num-corr) / total_num

    return err

def calc_acc(pred, lab):
    err = calc_err(pred, lab)
    return 1.0 - err

class UncertaintyLoss(nn.Module):
    """ Uncertainty loss for balancing multi-task learning dynamically """
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.log_sigma_se = nn.Parameter(torch.tensor(1.0))  # Learnable uncertainty parameter for SE
        self.log_sigma_ser = nn.Parameter(torch.tensor(1.0))  # Learnable uncertainty parameter for SER

    def forward(self, loss_se, loss_ser):
        # Compute weighted loss with learned uncertainty
        weighted_loss_se = loss_se / (2 * torch.exp(self.log_sigma_se) ** 2) + self.log_sigma_se
        weighted_loss_ser = loss_ser / (2 * torch.exp(self.log_sigma_ser) ** 2) + self.log_sigma_ser
        return weighted_loss_se + weighted_loss_ser
    
def kl_load_balancing_loss(gate_probs, alpha=0.01):
    """
    KL divergence between expert usage and uniform distribution.
    
    Args:
        gate_probs: Tensor of shape (batch, seq_len, num_experts)
    
    Returns:
        Scalar load balancing loss.
    """
    # Create hard routing mask from gate_probs (top-1 selection)
    indices = torch.argmax(gate_probs, dim=-1, keepdim=True)  # (B, T, 1)
    gate_mask = torch.zeros_like(gate_probs).scatter_(-1, indices, 1)  # (B, T, E)

    # Compute expert usage frequency (actual routing)
    expert_usage = gate_mask.float().mean(dim=(0, 1))  # (E,)

    # Ideal uniform distribution
    num_experts = gate_probs.shape[-1]
    ideal = torch.full_like(expert_usage, 1.0 / num_experts)

    # KL divergence: actual → ideal
    loss = F.kl_div(expert_usage.log(), ideal, reduction="batchmean")
    loss = alpha*loss
    return loss

def switch_transformer_load_balancing_loss(gate_probs, alpha=0.01):
    """
    Load balancing loss from the Switch Transformer paper.

    Args:
        gate_probs: Tensor of shape (batch, seq_len, num_experts) — softmax output

    Returns:
        Scalar load balancing loss.
    """
    B, T, E = gate_probs.shape

    top1_indices = torch.argmax(gate_probs, dim=-1, keepdim=True)  # (B, T, 1)
    expert_mask = torch.zeros_like(gate_probs).scatter_(-1, top1_indices, 1)  # (B, T, E)

    router_prob = gate_probs.float().mean(dim=(0, 1))  # Shape: (num_experts,)
    # fᵢ: Fraction of tokens routed to each expert (from hard top-1 decisions)
    expert_fraction = expert_mask.float().mean(dim=(0, 1))  # Shape: (num_experts,)

    # L = α * N * Σ (fᵢ * Pᵢ)
    loss = alpha * E * torch.sum(expert_fraction * router_prob)
    return loss