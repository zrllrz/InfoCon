"""
Reference: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange


class VQ2Linear(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False, log_choice=True):
        super().__init__()
        self.n_e = n_e
        self.n_e_bucket = torch.zeros(size=(n_e,), dtype=torch.int32)
        self.e_dim = e_dim

        self.beta = beta
        self.legacy = legacy
        self.log_choice = log_choice

        self.embedding = nn.Embedding(n_e + 1, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        # z shape (bs, e_dim)
        z = z.contiguous()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d[:, :-1], dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.contiguous()

        if self.log_choice is True:
            v = torch.var(min_encoding_indices.to(dtype=torch.float32))
            return z_q, loss, min_encoding_indices, v
        else:
            return z_q, loss, min_encoding_indices, None

class VQ2(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False, log_choice=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.beta = beta
        self.legacy = legacy
        self.log_choice = log_choice

        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z, flatten_in=False, flatten_out=False):
        # z shape (bs, T, e_dim)
        z = z.contiguous()
        if flatten_in is False:
            z_flattened = z.view(-1, self.e_dim)  # (bs * T, e_dim)
        else:
            z_flattened = z

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d[:, :-1], dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        z_q = z_q.contiguous()

        if flatten_out is False:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[1])
        # (B, T)
        if self.log_choice is True:
            v = torch.var(min_encoding_indices.to(dtype=torch.float32))
            return z_q, loss, min_encoding_indices, v
        else:
            return z_q, loss, min_encoding_indices, None

    def index_select(self, indices, flatten_in=False):
        indices = indices.contiguous()
        if flatten_in is False:
            indices_flattened = indices.view(-1)  # (bs * T)
        else:
            indices_flattened = indices
        # print(indices_flattened.shape)
        # print(indices.shape)
        z_out = self.embedding(indices_flattened)
        # print(z_out.shape)
        z_out = z_out.view(indices.shape[0], indices.shape[1], self.e_dim)
        # print(z_out.shape)

        return z_out

    # softmax-based commitment loss
    # def get_soft_commit_loss(self, z_commit_soft, indices):
    #     z = z_commit_soft.contiguous()
    #     z_flattened = z.view(-1, self.e_dim)
    #     d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
    #         torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
    #         torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
    #     d_exp = torch.exp(d)
    #     d_soft_term = torch.log(torch.add(torch.sum(d_exp, dim=1), 1e-5))
    #     loss_soft_penalty = torch.sum(d_soft_term)
    #     return loss_soft_penalty


class VQ3(nn.Module):
    def __init__(self, n_e, e_dim, p_change_th=0.8, log_choice=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.p_change_th = p_change_th
        self.log_choice = log_choice

        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        print('self.embedding')
        print(self.embedding.weight)
        print(self.embedding.weight.shape)

    def forward(self, p_change):
        # p_change: (B, T)
        # when at (i_b, t), it indicates the probability of switching key state
        B, T = p_change.shape

        p_change = p_change.contiguous()

        # We do switch key state when prob is greater than the hyper threshold
        p_change_signal = (p_change >= self.p_change_th)
        p_change_signal[:, 0:1] = False  # Do not switch at the very beginning

        # For switching time step, the first choice prob is p_change, otherwise it is (1.0 - p_change)
        p_first = torch.where(p_change_signal, p_change, 1.0 - p_change)

        # For indice selection
        p_change_signal = p_change_signal.type(torch.int64)

        # choose out 'first choice'
        p_ind_first = torch.clip(
            torch.cumsum(p_change_signal, dim=1),
            min=0, max=self.n_e - 1
        )

        # And regretable 'second choice'
        # for changing places, 'second choice' is the former key state
        # For mataining places, 'second choice' is the aiming key state (key state as sub-goal !!!)
        p_ind_second = torch.clip(
            torch.where(p_change_signal == 1, p_ind_first - 1, p_ind_first + 1),
            min=0, max=self.n_e
        )

        p_ind_first_flattened = p_ind_first.view(-1)
        p_ind_second_flattened = p_ind_second.view(-1)

        # indices selection
        z_first_flattened = self.embedding(p_ind_first_flattened)
        z_second_flattened = self.embedding(p_ind_second_flattened)
        z_first = z_first_flattened.view(B, T, self.e_dim)
        z_second = z_second_flattened.view(B, T, self.e_dim)

        # weighting output
        z_out = p_first[..., None] * z_first + (1.0 - p_first[..., None]) * z_second

        if self.log_choice is True:
            v = torch.var(z_first.to(dtype=torch.float32))
            return z_out, v
        else:
            return z_out, None


class VQNeighbor(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False, log_choice=True, persistence=None):
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

    def forward(self, z, flatten_in=False, flatten_out=False):
        # z shape (bs, T, e_dim)
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
            torch.argmin(d[:, 0, :], dim=1).unsqueeze(1),
            min=0,
            max=(self.n_e - 1)
        )
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
            self.min_indices = torch.min(encoding_indices)
            v = torch.max(encoding_indices) - self.min_indices
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


    def get_explore_indices(self, input, indices):
        with torch.no_grad():
            input_flattened = input.view(-1)[:, None]
            expanded_indices = (indices.view(-1))[None].expand(self.n_e + 1, -1)
            mask = (expanded_indices == self.arranged_mask).to(input.dtype)

            label_cnt = mask.sum(-1, keepdim=True)  # number of data in every label
            label_sum = mask @ input_flattened  # sum in every label
            label_square_sum = mask @ (input_flattened * input_flattened)  # sum of square in every label

            label_mean = label_sum / label_cnt  # mean in every label
            label_var = label_square_sum / (label_cnt - 1.0) - (label_sum * label_sum) / (label_cnt * (label_cnt - 1.0))
            # (estimation of) variation in every label
            torch.nan_to_num_(label_mean)
            torch.nan_to_num_(label_var)

            # higher than (mean + std) is intolerable
            upper_bound = label_mean[indices] + torch.sqrt(label_var)[indices]
            explore_mask = (upper_bound <= input)

            # thus these key state latents will try to become next one
            explore_indices = torch.clip(indices + explore_mask, min=0, max=self.n_e - 1)

            return explore_indices
