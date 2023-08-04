"""
Reference: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import sys
sys.path.append("/home/rzliu/AutoCoT/src/module")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange

from module_util import MLP


class VQNeighbor2(nn.Module):
    def __init__(self, n_e, e_dim, legacy_cluster=0.2):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.legacy_cluster = legacy_cluster
        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.key_to_qe = MLP(input_dim=e_dim, output_dim=e_dim, hidden_dims=[256, 256])

        self.register_buffer('arranged_mask', torch.arange(n_e + 1)[:, None])

    def select_from_index(self, indices):
        # indices (B, T)
        B, T = indices.shape
        indices = indices.contiguous()
        indices_flattened = indices.view(-1)  # (bs * T)
        key_hard = self.embedding(indices_flattened).view(B, T, self.e_dim)
        return key_hard

    def get_key_soft_indices(self, key_soft, zero_ts_mask=None):
        # key_soft: (B, T, self.e_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_flattened = key_soft.view(-1, self.e_dim)  # (bs * T, e_dim)

        # choosing indices does not need gradient
        with torch.no_grad():
            d = torch.sum(key_soft_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', key_soft_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
            d = d.view(B, T, -1)  # reshape the distance back to (B, T, n_e)

            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmin(d[:, 0, :], dim=1),
                min=0,
                max=(self.n_e - 1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T - 1, 1):
                d_t = d[:, (t + 1), :]

                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=d_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=d_t, dim=1, index=ind_next)

                d_choice_mask = torch.less_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        key_hard = self.embedding(encoding_indices.view(-1)).view(key_soft.shape)
        key_hard.contiguous()

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)
        return encoding_indices, key_hard, v

    def forward(self, key_soft, zero_ts_mask=None, preserve_grad=True):
        # key_soft shape (bs, T, self.e_dim)
        # zero_ts_mask (bs)

        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_flattened = key_soft.view(-1, self.e_dim)  # (bs * T, e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(key_soft_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight ** 2, dim=1) \
            - 2 * torch.einsum('bd,dn->bn', key_soft_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        with torch.no_grad():
            min_indices = torch.argmin(d, dim=1)

            d = d.view(B, T, self.n_e + 1)
            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmin(d[:, 0, :], dim=1),
                min=0,
                max=(self.n_e - 1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T-1, 1):
                d_t = d[:, (t + 1), :]
                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=d_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=d_t, dim=1, index=ind_next)
                d_choice_mask = torch.less_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        encoding_indices_flattened = encoding_indices.view(-1)
        key_hard_here = self.embedding(encoding_indices_flattened).view(key_soft.shape)
        key_hard_next = self.embedding(torch.clip(encoding_indices_flattened + 1,
                                                  min=0, max=self.n_e-1)).view(key_soft.shape)
        key_min = self.embedding(min_indices).view(key_soft.shape)

        # cluster choice
        loss_here_base = \
            torch.sum((key_soft.detach() - key_hard_here) ** 2, dim=-1) \
            + torch.sum((key_soft - key_hard_here.detach()) ** 2, dim=-1) * self.legacy_cluster
        loss_next_base = \
            torch.sum((key_soft.detach() - key_hard_next) ** 2, dim=-1) \
            + torch.sum((key_soft - key_hard_next.detach()) ** 2, dim=-1) * self.legacy_cluster
        loss_next_reg = torch.exp(-loss_here_base)
        loss_here_reg = torch.exp(-loss_next_base)

        # energy
        q_soft = self.key_to_qe(key_soft)
        q_hard_here = self.key_to_qe(key_hard_here)
        q_hard_next = self.key_to_qe(key_hard_next)

        energy = \
            (torch.sum((q_soft.detach() - q_hard_next) ** 2, dim=-1)
             - torch.sum((q_soft.detach() - q_hard_here) ** 2, dim=-1)) \
            + (torch.sum((q_soft - q_hard_next.detach()) ** 2, dim=-1)
               - torch.sum((q_soft - q_hard_here.detach()) ** 2, dim=-1)) * self.legacy_cluster

        # descend energy regularization
        encoding_indices_change = (encoding_indices[:, 1:] - encoding_indices[:, :-1]).to(dtype=torch.bool)
        energy_change = energy[:, 1:] - energy[:, :-1]
        same_hard_mask = torch.where(encoding_indices_change, 0.0, 1.0)
        energy_change = energy_change * same_hard_mask
        loss_energy_descent = \
            torch.maximum(energy_change + 1e-6 / self.n_e, torch.zeros_like(energy_change)).mean()

        # mean energy value, it will make the gradient include in information from pass and future states, actions
        # but it may not be a bad idea :)
        energy = energy.mean()

        loss_min_indices = \
            torch.sum((key_soft.detach() - key_min) ** 2, dim=-1) \
            + torch.sum((key_soft - key_min.detach()) ** 2, dim=-1) * self.legacy_cluster
        loss_min_here = torch.where(torch.less(loss_min_indices, loss_here_base), loss_min_indices, 0.0)
        loss_min_next = torch.where(torch.less(loss_min_indices, loss_next_base), loss_min_indices, 0.0)
        loss_here = loss_here_base + loss_here_reg - loss_min_here
        loss_next = loss_next_base + loss_next_reg - loss_min_next

        # preserve gradients
        if preserve_grad:
            key_hard = key_soft + (key_hard_here - key_soft).detach()
        key_hard = key_hard.contiguous()

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)

        return \
            key_hard, encoding_indices, v, \
            loss_here, loss_next, energy, loss_energy_descent


class VQNeighbor(nn.Module):
    def __init__(self, n_e, e_dim,
                 legacy_cluster=0.2,
                 legacy_energy=0.2, coe_structure=0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.legacy_cluster = legacy_cluster
        self.legacy_energy = legacy_energy
        self.coe_structure = coe_structure

        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.register_buffer('arranged_mask', torch.arange(n_e + 1)[:, None])

    def select_from_index(self, indices):
        # indices (B, T)
        B, T = indices.shape
        indices = indices.contiguous()
        indices_flattened = indices.view(-1)  # (bs * T)
        key_hard = self.embedding(indices_flattened).view(B, T, self.e_dim)
        return key_hard

    def get_loss_structure(self):
        # print('in get_loss_structure')
        d = torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        # print('d.requires_grad:', d.requires_grad, 'd.shape:', d.shape)

        # distance should be larger...
        # d_logit = torch.exp(torch.subtract(input=torch.zeros_like(d), other=d))
        # d_logit = torch.where(torch.isinf(d_logit), torch.zeros_like(d_logit), d_logit)
        # loss_dispersion = d_logit.sum() / (self.n_e * (self.n_e - 1))

        # should be in the range
        loss_range = \
            torch.maximum(torch.abs(self.embedding.weight) - 1.0 / self.n_e,
                          torch.zeros_like(self.embedding.weight)).mean()
        return loss_range

    def get_energy(self, key_soft, indices):
        # key_soft (B, T, self.e_dim)
        # indices (B, T)
        key_soft = key_soft.contiguous()

        key_hard_here = self.select_from_index(indices)
        key_hard_next = self.select_from_index(indices + 1)

        key_energy_mat = \
            (torch.sum((key_soft - key_hard_next.detach()) ** 2, dim=-1)
             - torch.sum((key_soft - key_hard_here.detach()) ** 2, dim=-1)) * self.legacy_energy + \
            (torch.sum((key_soft.detach() - key_hard_next) ** 2, dim=-1)
             - torch.sum((key_soft.detach() - key_hard_here) ** 2, dim=-1))
        # print('in get_energy, key_energy_mat.shape', key_energy_mat.shape)

        # descend energy regularization
        indices_change = (indices[:, 1:] - indices[:, :-1]).to(dtype=torch.bool)
        key_energy_change = key_energy_mat[:, 1:] - key_energy_mat[:, :-1]
        same_hard_mask = torch.where(indices_change, 0.0, 1.0)
        key_energy_change = key_energy_change * same_hard_mask
        loss_key_energy_descent = \
            torch.maximum(key_energy_change + 1e-6 / self.n_e, torch.zeros_like(key_energy_change)).mean()

        return key_energy_mat, loss_key_energy_descent

    def get_key_soft_indices(self, key_soft, zero_ts_mask=None):
        # key_soft: (B, T, self.e_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_flattened = key_soft.view(-1, self.e_dim)  # (bs * T, e_dim)

        # choosing indices does not need gradient
        with torch.no_grad():
            d = torch.sum(key_soft_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', key_soft_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
            d = d.view(B, T, -1)  # reshape the distance back to (B, T, n_e)

            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmin(d[:, 0, :], dim=1),
                min=0,
                max=(self.n_e - 1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T - 1, 1):
                d_t = d[:, (t + 1), :]

                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=d_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=d_t, dim=1, index=ind_next)

                d_choice_mask = torch.less_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        key_hard = self.embedding(encoding_indices.view(-1)).view(key_soft.shape)
        key_hard = key_soft + (key_hard - key_soft).detach()
        key_hard.contiguous()

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)
        return encoding_indices, key_hard, v

    def forward(self, key_soft, zero_ts_mask=None):
        # key_soft shape (bs, T, self.e_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_flattened = key_soft.view(-1, self.e_dim)  # (bs * T, e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(key_soft_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight ** 2, dim=1) \
            - 2 * torch.einsum('bd,dn->bn', key_soft_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        with torch.no_grad():
            min_indices = torch.argmin(d, dim=1)

            d = d.view(B, T, self.n_e + 1)
            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmin(d[:, 0, :], dim=1),
                min=0,
                max=(self.n_e - 1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T-1, 1):
                d_t = d[:, (t + 1), :]
                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=d_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=d_t, dim=1, index=ind_next)
                d_choice_mask = torch.less_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        encoding_indices_flattened = encoding_indices.view(-1)
        key_hard_here = self.embedding(encoding_indices_flattened).view(key_soft.shape)
        key_hard_next = self.embedding(encoding_indices_flattened + 1).view(key_soft.shape)
        key_min = self.embedding(min_indices).view(key_soft.shape)

        # Calculate energy at every point
        key_em_here = \
            torch.sum((key_soft.detach() - key_hard_here) ** 2, dim=-1) \
            + torch.sum((key_soft - key_hard_here.detach()) ** 2, dim=-1) * self.legacy_energy
        key_em_next = \
            torch.sum((key_soft.detach() - key_hard_next) ** 2, dim=-1) \
            + torch.sum((key_soft - key_hard_next.detach()) ** 2, dim=-1) * self.legacy_energy

        key_energy_mat = key_em_next - key_em_here

        # descend energy regularization
        encoding_indices_change = (encoding_indices[:, 1:] - encoding_indices[:, :-1]).to(dtype=torch.bool)
        key_energy_change = key_energy_mat[:, 1:] - key_energy_mat[:, :-1]
        same_hard_mask = torch.where(encoding_indices_change, 0.0, 1.0)
        key_energy_change = key_energy_change * same_hard_mask
        loss_key_energy_descent = \
            torch.maximum(key_energy_change + 1e-6 / self.n_e, torch.zeros_like(key_energy_change)).mean()

        # Clustering (Normal)
        loss_min_indices = \
            torch.sum((key_soft.detach() - key_min) ** 2, dim=-1) \
            + torch.sum((key_soft - key_min.detach()) ** 2, dim=-1) * self.legacy_cluster
        reg_persist_mat = torch.exp(-key_em_next)
        e_normal_mat = \
            torch.where(torch.greater(key_em_here, loss_min_indices - 1e-6 / self.n_e),
                        key_em_here - loss_min_indices + 1e-6 / self.n_e,
                        key_em_here) \
            + reg_persist_mat

        # Escape (Abnormal)
        reg_escape_mat = torch.exp(-key_em_here)
        e_abnormal_mat = key_em_next + reg_escape_mat

        # preserve gradients
        key_hard = key_soft + (key_hard_here - key_soft).detach()
        key_hard = key_hard.contiguous()

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)

        return \
            key_hard, encoding_indices, v, \
            loss_key_energy_descent, \
            key_energy_mat, e_normal_mat, e_abnormal_mat

    def label_mean(self, loss_criteria, indices):
        # loss_criteria: (B, T) size tensor of loss
        # indices: (B, T) size tensor of selected indices
        with torch.no_grad():
            loss_criteria_flattened = loss_criteria.view(-1)[:, None]
            expanded_indices = (indices.view(-1))[None].expand(self.n_e + 1, -1)
            mask = torch.eq(expanded_indices, self.arranged_mask).to(loss_criteria.dtype)

            label_cnt = mask.sum(-1, keepdim=True)  # number of data in every label
            label_sum = mask @ loss_criteria_flattened  # sum in every label
            label_mean = torch.div(label_sum, label_cnt).view(-1)  # mean in every label
            torch.nan_to_num_(label_mean)
            loss_mean_mat = label_mean[indices]

        return label_mean, loss_mean_mat


class VQElastic(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False, log_choice=True, persistence=None,
                 coe_loss_tolerance=1.0, coe_rate_tolerance=0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.beta = beta
        self.legacy = legacy
        self.log_choice = log_choice
        self.min_indices = -1

        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.register_buffer('arranged_mask', torch.arange(n_e + 1)[:, None])

        # persistence: use for long time policy
        if persistence is not None:
            # should be a float point between (0.0, 1.0)
            self.persistence = persistence
        else:
            self.persistence = 0.0


        # elastic table
        # for every codebook, they have three states
        # flinch(1): trying to shrink their range of guidance
        # explore(2): trying to expand their range of guidance
        # steady(0)
        # It will depend on the anomaly loss (too big) from statistic result of
        # action prediction loss from a label
        self.coe_loss_tolerance = coe_loss_tolerance
        self.coe_rate_tolerance = coe_rate_tolerance
        self.flinch_bound = coe_rate_tolerance / (1.0 + coe_loss_tolerance)
        self.explore_bound = 0.5 * coe_rate_tolerance / (1.0 + coe_loss_tolerance)

    def elastic_update(self, loss_criteria, indices):
        # loss_criteria: (B, T) size tensor of loss
        # indices: (B, T) size tensor of selected indices
        with torch.no_grad():
            loss_criteria_flattened = loss_criteria.view(-1)[:, None]
            expanded_indices = (indices.view(-1))[None].expand(self.n_e + 1, -1)
            mask = torch.eq(expanded_indices, self.arranged_mask).to(loss_criteria.dtype)

            label_cnt = mask.sum(-1, keepdim=True)  # number of data in every label
            label_sum = mask @ loss_criteria_flattened  # sum in every label
            label_mean = torch.div(label_sum, label_cnt).view(-1)  # mean in every label
            torch.nan_to_num_(label_mean)

            loss_tolerance_range = (1.0 + self.coe_loss_tolerance) * label_mean[indices]
            loss_anomaly_mask = torch.less(loss_tolerance_range, loss_criteria)
            expanded_loss_anomaly_mask = (loss_anomaly_mask.view(-1))[None].expand(self.n_e + 1, -1)
            expanded_loss_anomaly_mask = torch.logical_and(expanded_loss_anomaly_mask, mask).to(loss_criteria.dtype)

            label_anomaly_cnt = expanded_loss_anomaly_mask.sum(-1, keepdim=True)
            label_anomaly_rate = torch.div(label_anomaly_cnt, label_cnt).view(-1)
            torch.nan_to_num_(label_anomaly_rate)

            flinch_mask = \
                torch.where(torch.greater(label_anomaly_rate, self.flinch_bound), 1, 0)
            flinch_mask[self.n_e - 1] = 0  # last key should not flinch (it does not have a successive key to flinch!!!)
            explore_mask = \
                torch.where(torch.less(label_anomaly_rate, self.explore_bound), 1, 0)
            explore_mask = \
                torch.cat(
                    [torch.zeros(size=(1,), dtype=torch.int64, device=loss_criteria.device), explore_mask[: -1]],
                    dim=0
                )
        return flinch_mask, explore_mask

    def forward(self, z, flatten_in=False, flatten_out=False, loss_criteria=None, zero_ts_mask=None):
        # z shape (bs, T, e_dim)
        # assert loss_criteria is not None
        # assert zero_ts_mask is not None
        B, T = z.shape[0], z.shape[1]

        z = z.contiguous()
        if flatten_in is False:
            z_flattened = z.view(-1, self.e_dim)  # (bs * T, e_dim)
        else:
            z_flattened = z

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        d = d.view(B, T, -1)  # reshape the distance back to (B, T, n_e)

        # First timestep (0 step) in every context will use the nearest key
        encoding_indices = torch.clip(
            torch.argmin(d[:, 0, :], dim=1),
            min=0,
            max=(self.n_e - 1)
        )
        if zero_ts_mask is not None:
            encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
        encoding_indices = encoding_indices.unsqueeze(1)

        coe_persistence = torch.zeros(size=(B, 1), dtype=torch.float32, device=d.device)
        # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
        # e.g. when your hard key is #1 at context timestep 0,
        #      your hard key is from {#1, #2} at context timestep 1,
        for t in range(0, T-1, 1):
            d_t = d[:, (t + 1), :]

            ind_here = encoding_indices[:, t:(t + 1)]
            d_here = torch.gather(input=d_t, dim=1, index=ind_here)
            ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
            d_next = torch.gather(input=d_t, dim=1, index=ind_next)
            d_choice_mask = torch.less_equal(d_here, d_next - coe_persistence)
            ind_new = torch.where(d_choice_mask, ind_here, ind_next)
            coe_persistence = torch.where(d_choice_mask, coe_persistence + self.persistence / float(self.n_e), 0.0)
            encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        # ADJUST encoding_indices ACCORDING TO self.elastic_state
        if loss_criteria is not None:
            flinch_mask, explore_mask = self.elastic_update(loss_criteria, encoding_indices)
            flinch_update = flinch_mask[encoding_indices]
            explore_update = explore_mask[encoding_indices]
            # print(type(flinch_update), type(explore_update))
            encoding_indices = torch.clip(encoding_indices + flinch_update - explore_update, min=0, max=(self.n_e-1))

        z_q = self.embedding(encoding_indices.view(-1)).view(z.shape)

        # loss for embedding
        if not self.legacy:
            # loss = \
            #     self.beta * torch.mean((z_q.detach() - z) ** 2) + \
            #     torch.mean((z_q - z.detach()) ** 2)
            loss = \
                self.beta * self.get_loss_contrast(z, encoding_indices, self.embedding.weight.data.detach()) + \
                self.get_loss_contrast(z.detach(), encoding_indices, self.embedding.weight.data)

        else:
            # loss = \
            #     torch.mean((z_q.detach() - z) ** 2) + \
            #     self.beta * torch.mean((z_q - z.detach()) ** 2)
            loss = \
                self.get_loss_contrast(z, encoding_indices, self.embedding.weight.data.detach()) + \
                self.beta * self.get_loss_contrast(z.detach(), encoding_indices, self.embedding.weight.data)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.contiguous()

        if flatten_out is False:
            encoding_indices = encoding_indices.reshape(z_q.shape[0], z_q.shape[1])  # (B, T)

        if self.log_choice is True:
            self.min_indices, _ = torch.min(encoding_indices, dim=1)
            max_indices, _ = torch.max(encoding_indices, dim=1)
            v = torch.max(max_indices - self.min_indices)
            # v = torch.max(encoding_indices) - self.min_indices
            return z_q, loss, encoding_indices, v
        else:
            return z_q, loss, encoding_indices, None

    def select_from_index(self, indices, flatten_in=False):
        indices = indices.contiguous()
        if flatten_in is False:
            indices_flattened = indices.view(-1)  # (bs * T)
        else:
            indices_flattened = indices
        z_out = self.embedding(indices_flattened)
        z_out = z_out.view(indices.shape[0], indices.shape[1], self.e_dim)

        return z_out

    def get_loss_contrast(self, z, ind, book):
        z_q = book[ind]
        loss_indices = torch.sum((z - z_q) ** 2, dim=-1)
        loss_all = torch.sum((z.unsqueeze(2) - book.view(1, 1, -1, self.e_dim)) ** 2, dim=-1)
        loss_contrast = loss_indices.unsqueeze(-1) - loss_all + (1E-6 / self.n_e)
        loss_contrast = torch.maximum(loss_contrast, torch.zeros_like(loss_contrast))
        return loss_contrast.mean()
