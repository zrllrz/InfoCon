"""
Reference: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange


class VQ(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')


class VQStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(
            codebook, dim=0,
            index=indices_flatten
        )
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = grad_output.contiguous().view(-1, embedding_size)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_codebook


class VQEmbeddingLinear(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        # z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(
            self.embedding.weight,
            dim=0, index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class VQKeyState(nn.Module):
    def __init__(self, n_keys, key_dim, len_prompt):
        super().__init__()
        self.key_book = nn.Embedding(n_keys, key_dim)
        self.key_book.weight.data.uniform_(-1./n_keys, 1./n_keys)

        self.n_keys = n_keys          # number of possible key states
        self.key_dim = key_dim        # dim of embedding key states
        self.content_dim = key_dim * len_prompt  # flatten prediction of the prompt into the Act Net

    def forward(self, key_emb):
        key_emb_ = key_emb.contiguous()
        indices = vq(key_emb_, self.key_book.weight)
        return indices

    def straight_through(self, key_emb):
        key_emb_ = key_emb.contiguous()
        key_q_, indices = vq_st(key_emb_, self.key_book.weight.detach())
        key_q = key_q_.contiguous()

        key_q_bar_ = torch.index_select(
            self.key_book.weight,
            dim=0, index=indices
        )
        key_q_bar = key_q_bar_.contiguous()

        return key_q, key_q_bar


class VQ2Linear(nn.Module):
    def __init__(self, n_e, e_dim, beta, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.beta = beta
        print('in init of VQ2Linear', self.beta, type(self.beta))
        self.legacy = legacy

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

        return z_q, loss


vq = VQ.apply
vq_st = VQStraightThrough.apply
__all__ = [vq, vq_st]


