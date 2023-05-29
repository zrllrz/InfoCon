"""
Reference: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange


class VQ2Linear(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False, log_choice=True):
        super().__init__()
        self.n_e = n_e
        self.n_e_bucket = torch.zeros(size=(n_e,), dtype=torch.int32)
        self.e_dim = e_dim

        self.beta = beta
        print('in init of VQ2Linear', self.beta, type(self.beta))
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

        perplexity = None
        min_encodings = None

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
            # print('Try to log the time each indice has been selected')
            # print('but it will drawback training efficiency')
            # print('so we try to calculate variation instead')
            v = torch.var(min_encoding_indices.to(dtype=torch.float32))
            return z_q, loss, min_encoding_indices, v
        else:
            return z_q, loss, min_encoding_indices, None
