import torch
import torch.nn
import torch.nn.functional as F


def anomaly_score(r, sharpen=100.0):
    f = torch.abs(r - 1.0)
    f = ((r - 1.0) + f) * 0.5
    return (f / r) ** (1.0 / sharpen)

def cos_anomaly_score(vcos, sharpen=10.0):
    f = vcos * sharpen
    f = (torch.abs(torch.tanh(f)) - torch.tanh(f)) * 0.5
    return f


def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses), losses
    else:
        assert losses.shape == weights.shape, losses.shape
        return torch.mean(losses * weights), losses


def get_loss(preds, targets, lengths):
    # If we have sequences of varied lengths, use masks so we do not compute loss
    # over padded values. If we set max_seq_length=min_seq_length, then it should
    # not matter since all sequences have the same length.
    B = preds.shape[0]
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    lengths = lengths[:, None]  # B x 1
    temp = torch.arange(0, max_len)[None].expand(B, -1).cuda()  # B x max_len
    masks = torch.less(temp, lengths.expand(B, max_len)).float()  # B x max_len
    # (temp < lengths.expand(B, max_len)).float()  # B x max_len

    loss, lossess = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss, lossess.reshape(B, -1)


def init_centroids(datas, n_centroids):
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    cent_init = datas[i0][None, :]
    for _ in range(n_centroids - 1):
        d = torch.sum(datas ** 2, dim=1, keepdim=True) + \
            torch.sum(cent_init ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', datas, rearrange(cent_init, 'n d -> d n'))

        d_min = d.min(dim=1)[0] + 1e-5
        d_min = torch.where(torch.isnan(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.less(d_min, 0.0), torch.zeros_like(d_min), d_min)
        d_min_sum = d_min.sum()

        if d_min_sum == 0.0 or torch.isnan(d_min_sum) or torch.isinf(d_min_sum):
            i = torch.randint(0, N, (1,))[0]
        else:
            i = d_min.multinomial(num_samples=1)[0]

        cent_init = torch.cat([cent_init, datas[i][None, :]], dim=0)
    return cent_init


def init_centroids_neighbor(datas, unified_t, n_centroids):
    # datas: (B*T, e_dim), key_soft
    # unified_t: (B*T), unified timesteps
    # n_centroids: number of centroids
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    unified_t_unsq = unified_t.view(-1, 1)
    cent_init_ind = torch.tensor([i0])
    cent_init_u_t = unified_t[i0].view(1)
    for _ in range(n_centroids - 1):
        d = torch.abs(unified_t_unsq - cent_init_u_t)
        d_min = d.min(dim=1)[0] + 1e-5
        d_min = torch.where(torch.isnan(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.less(d_min, 0.0), torch.zeros_like(d_min), d_min)
        d_min_sum = d_min.sum()

        if d_min_sum == 0.0 or torch.isnan(d_min_sum) or torch.isinf(d_min_sum):
            i = torch.randint(0, N, (1,))[0]
        else:
            i = d_min.multinomial(num_samples=1)[0]
        cent_init_ind = torch.cat([cent_init_ind, torch.tensor([i])], dim=0)
        cent_init_u_t = torch.cat([cent_init_u_t, unified_t_unsq[i]], dim=0)


    # sorted by unified time step!!!!!!
    _, sorted_sub_ind = torch.sort(cent_init_u_t)
    cent_init_ind = cent_init_ind[sorted_sub_ind.to(cent_init_ind.device)]
    cent_init_datas = datas[cent_init_ind]

    return cent_init_datas
