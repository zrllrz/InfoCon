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

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # z shape (bs, e_dim)
        z = z.contiguous()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) -2 * \
            torch.einsum('bd,dn->bn', z, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)

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

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

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

        min_encoding_indices = torch.argmin(d, dim=1)

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



