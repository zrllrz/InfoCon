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

from module_util import MLP, TimeSphereEncoder, FreqEncoder


# def init_ut(delta_t):
#     ut_cumsum = torch.cumsum(delta_t, dim=0)
#     ut = torch.div(ut_cumsum, ut_cumsum[-1, 0] + 1e-12)
#     return ut

class increaseEncourageIdentityMap(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_increase = torch.less_equal(grad_output, 0.0)
        grad_scale = torch.where(mask_increase, 10.0, 1.0)
        return grad_output * grad_scale

increae_encourage_identity_map = increaseEncourageIdentityMap.apply


class VQClassifierNNTime(nn.Module):
    def __init__(self, key_dim, n_e, e_dim, e_split, KT=0.1, use_ema=False, coe_ema=0.5,
                 use_ft_emb=True, use_st_emb=True, t_emb_rate=1.2,
                 use_prob_sel_train=False, use_timestep_appeal=False):
        super().__init__()

        assert (not (use_ft_emb and use_st_emb))

        self.key_dim = key_dim
        self.n_e = n_e
        self.e_dim = e_dim
        self.e_split = e_split  # for later HN, how many segment do we have, we normalize vparams below based on e_split
        self.KT = KT

        self.sm = nn.Softmax(dim=-1)

        # length of every prototype
        self.r_keys = nn.Embedding(n_e, 1)
        nn.init.constant_(self.r_keys.weight, 0.999)
        self.r_activate = nn.Hardtanh(min_val=0.0, max_val=1.0)

        arange = torch.arange(n_e, dtype=torch.float32)
        # along with scaler that transform them into different time area
        t_base = torch.div(arange, n_e)
        self.register_buffer('t_base', t_base.unsqueeze(-1))
        if use_st_emb:
            self.t_emb = TimeSphereEncoder(rate=t_emb_rate)
        else:
            self.t_emb = FreqEncoder(half_t_size=4, feature_size=key_dim)

        # control time area of prototype
        self.t_keys = nn.Embedding(n_e, 1)
        delta_t = torch.div(torch.rand(size=(n_e, 1)), n_e)
        print('delta_t.shape', delta_t.shape)
        print('t_base.shape', t_base.shape)
        ut = delta_t + t_base.unsqueeze(-1)
        print('init ut', ut.squeeze(-1))
        self.t_keys.weight.data = ut

        self.keys = nn.Embedding(n_e, key_dim)
        self.keys.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        self.keys.weight.data = F.normalize(self.keys.weight.data, p=2.0, dim=-1)

        if use_ft_emb:
            self.keys.weight.data = self.t_emb(self.keys.weight.data, ut.squeeze(-1))

        self.use_prob_sel_train = use_prob_sel_train
        self.use_timestep_appeal = use_timestep_appeal

        # parameters
        self.vparams = nn.Embedding(n_e, e_dim)
        self.vparams.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.use_ema = use_ema
        self.coe_ema = coe_ema
        # None or a float point bound (how to calculate it will be defined in autocot model...)

        self.register_buffer('arange', arange)
        self.register_buffer('not_same', 1.0 - torch.eye(n_e))
        mask_adj = torch.abs(torch.sub(self.arange.view(-1, 1), self.arange))
        mask_adj = torch.le(mask_adj, 1)
        print('mask_adj\n', mask_adj)
        self.register_buffer('mask_adj', mask_adj)

        self.use_st_emb = use_st_emb
        self.use_ft_emb = use_ft_emb
        if use_st_emb:
            self.key_dim_t_emb = key_dim + 1
        else:
            self.key_dim_t_emb = key_dim

    def get_keys(self):
        # for the reason that our keys and its time embedding is separated
        # we provide a method that transform this easily
        keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)
        u_t = self.t_keys.weight  # (n_e, 1)
        if self.use_st_emb:
            keys_norm_t_emb = self.t_emb(keys_norm, u_t.squeeze(-1))
        else:
            keys_norm_t_emb = keys_norm
        return keys_norm_t_emb

    def get_r(self):
        r = self.r_activate(self.r_keys.weight)
        return r  # (n_e, 1)

    def reset_oor_r(self, left_step=10_000):
        with torch.no_grad():
            # First deal with the clipping
            # then dealing with value out of range [0.001, 0.999]
            m_999 = torch.greater(self.r_keys.weight.data, 0.999)
            m_1 = torch.less(self.r_keys.weight.data, 0.1)

            self.r_keys.weight.data = torch.where(m_1, 0.1, self.r_keys.weight.data)
            self.r_keys.weight.data = torch.where(m_999, 0.999, self.r_keys.weight.data)

    def split_vparams_norm(self, vparams):
        # (..., vparams), this part, the vparams may be weighed vparams (vparams_w)
        # so we need a params here...
        vparams_norm = []
        for i in range(self.e_split):
            i_begin = i * (self.e_dim // self.e_split)
            i_end = (i + 1) * (self.e_dim // self.e_split)
            vparams_norm.append(F.normalize(vparams[..., i_begin:i_end], p=2.0, dim=-1))
        vparams_norm = torch.cat(vparams_norm, dim=-1)
        return vparams_norm

    def score_to_weight(self, score):
        # score (..., n_e)
        weight = self.sm(torch.div(score, self.KT))  # (..., n_e)
        weight = torch.nan_to_num(weight)
        return weight

    def get_time_emb_keys(self, keys, u_t):
        # keys: (..., key_dim)
        # u_t: (...)
        keys = keys.contiguous()
        keys_norm = F.normalize(keys, p=2.0, dim=-1)
        if self.use_st_emb or self.use_ft_emb:
            keys_norm_t_emb = self.t_emb(feature=keys_norm, unified_t=u_t)  # key_soft: (..., key_dim_t_emb)
        else:
            keys_norm_t_emb = keys_norm
        return keys_norm_t_emb

    def get_key_soft_indices(self, key_soft, u_t):
        # key_soft: (B, T, key_dim)
        # choosing indices does not need gradient
        with torch.no_grad():
            key_soft = key_soft.contiguous()
            B, T = key_soft.shape[0], key_soft.shape[1]
            key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
            if self.use_st_emb or self.use_ft_emb:
                key_soft_t_emb = self.t_emb(feature=key_soft_norm, unified_t=u_t)  # key_soft: (B, T, key_dim + 1)
            else:
                key_soft_t_emb = key_soft_norm
            key_soft_flattened = key_soft_t_emb.view(B*T, self.key_dim_t_emb)  # (B*T, key_dim + 1)


            # construct usage keys in the codebook (time + feature)
            keys_norm = self.get_keys()  # (n_e, key_dim + 1)
            keys_norm = keys_norm * self.get_r()
            score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                               rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)
            # use the nearest key
            encoding_indices = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
            encoding_indices = encoding_indices.view(B, T)

            return encoding_indices

    def get_w_cnt(self, encoding_indices):
        B, T = encoding_indices.shape
        ei_flattened = encoding_indices.view(-1)
        sum_mask = torch.eq(self.arange.view(-1, 1).repeat(1, B*T),
                            ei_flattened.view(1, -1).repeat(self.n_e, 1))
        # print('sum_mask.shape', sum_mask.shape)
        cnt = torch.add(torch.sum(sum_mask, dim=-1), 1.0)
        w = torch.div(1.0, cnt)
        w = torch.nan_to_num(w)
        w_cnt = w[encoding_indices]
        return w_cnt

    def ema_forward(self, key_soft, u_t):
        # key_soft shape (B, T, self.key_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]

        # first construct key_soft with timestep embedding
        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)  # (B, T, key_dim)
        if self.use_st_emb or self.use_ft_emb:
            key_soft_t_emb = self.t_emb(feature=key_soft_norm, unified_t=u_t)  # (B, T, key_dim + 1)
        else:
            key_soft_t_emb = key_soft_norm
        key_soft_flattened = key_soft_t_emb.view(B*T, self.key_dim_t_emb)  # (B*T, key_dim + 1)

        # construct usage keys in the codebook (time + feature)
        keys_norm = self.get_keys()  # (n_e, key_dim + 1)
        # print('keys_norm.shape:', keys_norm.shape)
        keys_norm = keys_norm.detach()  # update of ema do not need gradient
        keys_norm = keys_norm * self.get_r()  # (n_e, key_dim + 1)
        # print('keys_norm.shape:', keys_norm.shape)

        # calc score_ksh
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)

        # Neareset Neighbor for indices and hard vparams
        encoding_indices_flattened = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
        # encoding_indices = encoding_indices_flattened.view(B, T)  # (B, T)

        # EMA update of keys behind
        with torch.no_grad():
            # u_t_flattened = u_t.view(-1)  # (B*T)
            sum_mask = torch.eq(self.arange.view(-1, 1).repeat(1, B * T),
                                encoding_indices_flattened.view(1, -1).repeat(self.n_e, 1))
            # if self.use_timestep_appeal:
            #     print('use_timestep_appeal')
            #     time_low_mask = \
            #         torch.greater_equal(u_t_flattened.view(1, -1).repeat(self.n_e, 1), self.t_base)
            #     time_high_mask = \
            #         torch.less(u_t_flattened.view(1, -1).repeat(self.n_e, 1), torch.add(self.t_base, 1.0 / self.n_e))
            #     time_mask = torch.logical_and(time_low_mask, time_high_mask)  # (n_e, B*T)
            #     sum_mask = torch.logical_or(sum_mask, time_mask)

            sum_mask = sum_mask.to(dtype=torch.float32)

            cnt = 1 + torch.sum(sum_mask, dim=-1, keepdim=True)  # (n_e, 1)
            mean_key_soft = torch.div(sum_mask @ key_soft_flattened, cnt)  # (n_e, key_dim + 1)
            mean_key_soft = torch.nan_to_num(mean_key_soft)  # (n_e, key_dim + 1)
            keys_norm_new = F.normalize(keys_norm * self.coe_ema + mean_key_soft * (1.0 - self.coe_ema),
                                        p=2.0, dim=-1)  # (n_e, key_dim + 1)
            if self.use_st_emb:
                t_keys_new, _, keys_new = self.t_emb.split_t_f(keys_norm_new)
                t_keys_new, index_switch = torch.sort(t_keys_new)  # swtich according to time order
                self.t_keys.weight.data = t_keys_new.unsqueeze(1)

                # switch other three books
                self.keys.weight.data = keys_new[index_switch]
                self.r_keys.weight.data = self.r_keys.weight.data[index_switch]
                self.vparams.weight.data = self.vparams.weight.data[index_switch]

                # switch encoding_indices
                encoding_indices_flattened = index_switch[encoding_indices_flattened]
            else:
                self.keys.weight.data = keys_norm_new

            encoding_indices = encoding_indices_flattened.view(B, T)  # (B, T)
            w_cnt = self.get_w_cnt(encoding_indices)

        # re-calculate score_ksh...
        # calc score_ksh
        keys_norm = self.get_keys()  # (n_e, key_dim + 1)
        keys_norm = keys_norm.detach()  # update of ema do not need gradient
        keys_norm = keys_norm * self.get_r()  # (n_e, key_dim + 1)
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)
        # score_ksh = score_ksh_flattened.view(B, T, self.n_e)  # (B, T, n_e)
        # transmit into weight
        w_flattened = self.score_to_weight(score_ksh_flattened)  # (B*T, n_e)

        # choose out prototype prob for each soft key (after update...)
        # if not self.use_prob_sel_train:
        #     w_max_flattened = torch.gather(w_flattened, dim=-1, index=encoding_indices_flattened.unsqueeze(-1)).squeeze(-1)
        #     w_max = w_max_flattened.view(B, T)
        if self.use_prob_sel_train:
            print('use_prob_sel_train')
            # we only choose the adjacent index
            # print('use_prob_sel_train')
            mask_adj = self.mask_adj[encoding_indices_flattened]
            score_ksh_flattened_mask = torch.where(mask_adj, score_ksh_flattened, float('-inf'))
            w_flattened_mask = self.score_to_weight(score_ksh_flattened_mask)
            # reset encoding_indices_flattened using prob sampling
            encoding_indices_flattened = w_flattened_mask.view(-1, self.n_e).multinomial(1).view(B*T)
            encoding_indices = encoding_indices_flattened.view(B, T)

        w_max_flattened = torch.gather(w_flattened, dim=-1, index=encoding_indices_flattened.unsqueeze(-1)).squeeze(-1)
        w_max = w_max_flattened.view(B, T)

        # print(w_max.shape)

        # to weighted params
        vparams_norm = self.split_vparams_norm(self.vparams.weight)  # (n_e, e_dim)
        vparams_w_flattened = \
            w_flattened.view(B * T, 1, self.n_e) @ (vparams_norm.view(1, self.n_e, self.e_dim).detach())
        key_w_flattened = \
            w_flattened.view(B * T, 1, self.n_e) @ (keys_norm.view(1, self.n_e, self.key_dim_t_emb).detach())

        vparams_w_flattened = vparams_w_flattened.squeeze(1)  # (B*T, e_dim)
        key_w_flattened = key_w_flattened.squeeze(1)  # (B*T, key_dim + 1)

        vparams_w_flattened = self.split_vparams_norm(vparams_w_flattened)  # (B*T, e_dim)
        key_w_flattened = F.normalize(key_w_flattened, p=2.0, dim=-1)  # (B*T, key_dim_t_emb)

        vparams_w = vparams_w_flattened.view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_w = vparams_w.contiguous()  # (B, T, e_dim)
        key_w = key_w_flattened.view(B, T, self.key_dim_t_emb)
        key_w = key_w.contiguous()

        vparams_hard = self.vparams(encoding_indices).view(B, T, self.e_dim)  # (B, T, e_dim)
        key_hard = keys_norm[encoding_indices.view(-1)].view(B, T, self.key_dim_t_emb)  # (B, T, key_dim+1)

        vparams_hard = self.split_vparams_norm(vparams_hard)  # normalization (B, T, e_dim)
        vparams_hard = vparams_hard.contiguous()
        key_hard = key_hard.contiguous()
        vparams_hard = vparams_w + (vparams_hard - vparams_w).detach()  # store gradient
        key_hard = key_w + (key_hard - key_w).detach()  # store gradient

        return \
            encoding_indices, key_hard, \
            vparams_w, vparams_hard, w_max, w_cnt, key_soft_t_emb

    def forward(self, key_soft, u_t):
        # key_soft shape (B, T, self.key_dim)
        assert self.use_ema, 'should not reach here'
        return self.ema_forward(key_soft, u_t)



class VQClassifierNN(nn.Module):
    def __init__(self, key_dim, n_e, e_dim, KT=1.0, use_ema=False, coe_ema=0.95):
        super().__init__()
        self.key_dim = key_dim
        self.n_e = n_e
        self.e_dim = e_dim
        self.KT = KT

        self.sm = nn.Softmax(dim=-1)

        self.keys = nn.Embedding(n_e, key_dim)
        self.keys.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.vparams = nn.Embedding(n_e, e_dim)
        self.vparams.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.use_ema = use_ema
        self.coe_ema = coe_ema

        arange = torch.arange(n_e, dtype=torch.float32)
        self.register_buffer('arange', arange)
        self.register_buffer('not_same', 1.0 - torch.eye(n_e))

        self.loss_label = torch.zeros(size=(n_e,), dtype=torch.float32)
        self.n_label = torch.zeros(size=(n_e,), dtype=torch.float32)
        self.min_ut = torch.full(size=(n_e,), fill_value=float('+inf'))
        self.max_ut = torch.full(size=(n_e,), fill_value=float('-inf'))

    def update_loss_label(self, losses, unified_t, indices):
        # losses (...), loss at each step
        # unified_t (...), unified time at each step
        # indices (...), alonging indices
        # always flattened
        with torch.no_grad():
            losses_flattened = losses.view(-1)  # (L)
            ut_flattened = unified_t.view(-1)  # (L)
            indices_flattened = indices.view(-1)  # (L)
            L = losses_flattened.shape[0]

            sum_mask = torch.eq(
                self.arange.view(-1, 1).repeat(1, L),
                indices_flattened.view(1, -1).repeat(self.n_e, 1)
            )  # (n_e, L)
            sum_mask_float = sum_mask.to(dtype=torch.float32)
            cnt = torch.sum(sum_mask_float, dim=-1)  # (n_e)
            sum_losses = (sum_mask_float @ losses_flattened.unsqueeze(-1)).squeeze(-1)  # (n_e)

            # update loss_label (average of label loss)
            self.loss_label = (self.loss_label * self.n_label + sum_losses) / (self.n_label + cnt)
            self.n_label += cnt

            # update min_ut and max_ut
            max_ut = torch.where(sum_mask, ut_flattened.view(1, -1).repeat(self.n_e, 1), float('-inf'))
            min_ut = torch.where(sum_mask, ut_flattened.view(1, -1).repeat(self.n_e, 1), float('+inf'))
            max_ut = torch.max(max_ut, dim=-1)[0]  # (n_e)
            min_ut = torch.min(min_ut, dim=-1)[0]  # (n_e)
            self.max_ut = torch.where(torch.greater(max_ut, self.max_ut), max_ut, self.max_ut)
            self.min_ut = torch.where(torch.less(min_ut, self.min_ut), min_ut, self.min_ut)

    def refresh_loss_label(self):
        # clean loss_label and n_label
        self.loss_label = torch.zeros_like(self.loss_label)
        self.n_label = torch.zeros_like(self.n_label)
        self.min_ut = torch.full_like(self.min_ut, fill_value=float('+inf'))
        self.max_ut = torch.full_like(self.min_ut, fill_value=float('-inf'))


    def loss_dispersion(self):
        keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)  # (n_e, key_dim)
        l_dispersion_k = torch.einsum('bd,dn->bn', keys_norm,
                                      rearrange(keys_norm, 'n d -> d n'))  # (n_e, n_e)
        l_dispersion_k = torch.exp(torch.div(l_dispersion_k, self.KT))   # (n_e, n_e)
        l_dispersion_k = l_dispersion_k * self.not_same  # (n_e, n_e)
        l_dispersion_k = torch.sum(l_dispersion_k, dim=-1)  # (n_e)
        l_dispersion_k = torch.div(l_dispersion_k, float(self.n_e - 1))  # (n_e)
        l_dispersion_k = torch.log(l_dispersion_k)  # (n_e)
        l_dispersion_k = l_dispersion_k.mean()

        vparams_norm = F.normalize(self.vparams.weight, p=2.0, dim=-1)  # (n_e, key_dim)
        l_dispersion_vp = torch.einsum('bd,dn->bn', vparams_norm,
                                       rearrange(vparams_norm, 'n d -> d n'))  # (n_e, n_e)
        l_dispersion_vp = torch.exp(torch.div(l_dispersion_vp, self.KT))  # (n_e, n_e)
        l_dispersion_vp = l_dispersion_vp * self.not_same  # (n_e, n_e)
        l_dispersion_vp = torch.sum(l_dispersion_vp, dim=-1)  # (n_e)
        l_dispersion_vp = torch.div(l_dispersion_vp, float(self.n_e - 1))  # (n_e)
        l_dispersion_vp = torch.log(l_dispersion_vp)  # (n_e)
        l_dispersion_vp = l_dispersion_vp.mean()

        return l_dispersion_k + l_dispersion_vp

    def score_to_weight(self, score):
        # score (..., n_e)
        weight = self.sm(torch.div(score, self.KT))  # (..., n_e)
        weight = torch.nan_to_num(weight)
        return weight


    def cos_sim(self, v_soft, mode='key'):
        if mode == 'key':
            key_soft = v_soft  # (B, T, key_dim)
            with torch.no_grad():
                B = key_soft.shape[0]
                key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
                keys = self.keys.weight
                keys_norm = F.normalize(keys, p=2.0, dim=-1)
                keys_norm = keys_norm.view(1, self.n_e, self.key_dim).repeat(B, 1, 1)

                k_cs_ss = torch.bmm(key_soft_norm, rearrange(key_soft_norm, 'b t c -> b c t'))
                k_cs_sh = torch.bmm(key_soft_norm, rearrange(keys_norm, 'b t c -> b c t'))

                return k_cs_ss, k_cs_sh
        elif mode == 'vparams':
            vparams_soft = v_soft  # (B, T, e_dim)
            with torch.no_grad():
                B = vparams_soft.shape[0]
                vparams_soft_norm = F.normalize(vparams_soft, p=2.0, dim=-1)
                vparams = self.vparams.weight
                vparams_norm = F.normalize(vparams, p=2.0, dim=-1)
                vparams_norm = vparams_norm.view(1, self.n_e, self.e_dim).repeat(B, 1, 1)

                vp_cs_ss = torch.bmm(vparams_soft_norm, rearrange(vparams_soft_norm, 'b t c -> b c t'))
                vp_cs_sh = torch.bmm(vparams_soft_norm, rearrange(vparams_norm, 'b t c -> b c t'))

                return vp_cs_ss, vp_cs_sh
        else:
            print('unknown cos_sim type')
            assert False

    def get_key_soft_indices(self, key_soft):
        # key_soft: (B, T, self.e_dim)
        # choosing indices does not need gradient
        with torch.no_grad():
            key_soft = key_soft.contiguous()
            B, T = key_soft.shape[0], key_soft.shape[1]
            key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
            key_soft_flattened = key_soft_norm.view(B*T, self.key_dim)  # (B*T, key_dim)

            keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)  # (n_e, key_dim)

            score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                               rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)

            # use the nearest key
            encoding_indices = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
            encoding_indices = encoding_indices.view(B, T)

            return encoding_indices

    def policy_forward(self, key_soft):
        # key_soft shape (B, T, key_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
        key_soft_flattened = key_soft_norm.view(B*T, self.key_dim)  # (B*T, key_dim)
        keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)  # (n_e, key_dim)
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)

        # to weighted params
        w_flattened = self.score_to_weight(score_ksh_flattened)  # (B*T, n_e)
        vparams_norm = F.normalize(self.vparams.weight, p=2.0, dim=-1)  # (n_e, e_dim)
        vparams_w_flattened = w_flattened.view(B * T, 1, self.n_e) @ vparams_norm.view(1, self.n_e, self.e_dim)
        vparams_w_flattened = vparams_w_flattened.squeeze(1)  # (B*T, e_dim)
        vparams_w_flattened = F.normalize(vparams_w_flattened, p=2.0, dim=-1)  # (B*T, e_dim)
        vparams_w = vparams_w_flattened.view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_w = vparams_w.contiguous()  # (B, T, e_dim)

        # Neareset Neighbor for indices and hard vparams
        encoding_indices = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
        encoding_indices = encoding_indices.view(B, T)  # (B, T)

        vparams_hard = self.vparams(encoding_indices).view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_hard = F.normalize(vparams_hard, p=2.0, dim=-1)  # normalization (B, T, e_dim)
        vparams_hard = vparams_hard.contiguous()
        vparams_hard = vparams_w + (vparams_hard - vparams_w).detach()  # store gradient

        return encoding_indices, vparams_w, vparams_hard

    def clustering_forward(self, key_soft):
        # key_soft shape (B, T, key_dim)
        # key_soft shape (B, T, key_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]

        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)  # (B, T, key_dim)
        key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)  # (B*T, key_dim)
        keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)  # (n_e, key_dim)
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)
        score_ksh = score_ksh_flattened.view(B, T, self.n_e)  # (B, T, n_e)

        # calc score_kss
        score_kss = torch.einsum('btc,bcs->bts', key_soft_norm,
                                 rearrange(key_soft_norm, 'b t c -> b c t'))  # (B, T, T)

        # to weighted params
        w_flattened = self.score_to_weight(score_ksh_flattened)  # (B*T, n_e)
        vparams_norm = F.normalize(self.vparams.weight, p=2.0, dim=-1)  # (n_e, e_dim)
        vparams_w_flattened = w_flattened.view(B*T, 1, self.n_e) @ vparams_norm.view(1, self.n_e, self.e_dim)
        vparams_w_flattened = vparams_w_flattened.squeeze(1)  # (B*T, e_dim)
        vparams_w_flattened = F.normalize(vparams_w_flattened, p=2.0, dim=-1)  # (B*T, e_dim)
        vparams_w = vparams_w_flattened.view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_w = vparams_w.contiguous()  # (B, T, e_dim)

        # calc score_vpss (cos sim between vparams_w)
        score_vpss = torch.einsum('btc,bcs->bts', vparams_w,
                                  rearrange(vparams_w, 'b t c -> b c t'))  # (B, T, T)

        # calc score_vpsh (cos sim between vparams_w and self.vparams)
        score_vpsh_flattened = torch.einsum('bd,dn->bn', vparams_w_flattened,
                                            rearrange(vparams_norm, 'n d -> d n'))  # (B*T, n_e)
        score_vpsh = score_vpsh_flattened.view(B, T, self.n_e)  # (B, T, n_e)


        # Neareset Neighbor for indices and hard vparams
        encoding_indices = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
        encoding_indices = encoding_indices.view(B, T)  # (B, T)

        vparams_hard = self.vparams(encoding_indices).view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_hard = F.normalize(vparams_hard, p=2.0, dim=-1)  # normalization (B, T, e_dim)
        vparams_hard = vparams_hard.contiguous()
        vparams_hard = vparams_w + (vparams_hard - vparams_w).detach()  # store gradient

        return \
            encoding_indices, vparams_w, vparams_hard, \
            score_vpss, score_vpsh, score_kss, score_ksh

    def ema_forward(self, key_soft):
        # key_soft shape (B, T, self.key_dim)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]

        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)  # (B, T, key_dim)
        key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)  # (B*T, key_dim)
        keys_norm = F.normalize(self.keys.weight.detach(), p=2.0, dim=-1)  # (n_e, key_dim)

        # calc score_ksh
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)

        # Neareset Neighbor for indices and hard vparams
        encoding_indices_flattened = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)
        encoding_indices = encoding_indices_flattened.view(B, T)  # (B, T)

        # EMA update of keys behind
        with torch.no_grad():
            sum_mask = torch.eq(
                self.arange.view(-1, 1).repeat(1, B * T),
                encoding_indices_flattened.view(1, -1).repeat(self.n_e, 1)
            ).to(dtype=torch.float32)
            cnt = torch.sum(sum_mask, dim=-1, keepdim=True)  # (n_e, 1)
            mean_key_soft = torch.div(sum_mask @ key_soft_flattened, cnt)  # (n_e, key_dim)
            mean_key_soft = torch.nan_to_num(mean_key_soft)  # (n_e, key_dim)
            self.keys.weight.data = F.normalize(
                self.keys.weight.data * self.coe_ema + mean_key_soft * (1.0 - self.coe_ema),
                p=2.0, dim=-1
            )
        # re-calculate score_ksh...
        # calc score_ksh
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(self.keys.weight.detach(), 'n d -> d n'))  # (B*T, n_e)
        score_ksh = score_ksh_flattened.view(B, T, self.n_e)  # (B, T, n_e)
        # transmit into weight
        w_flattened = self.score_to_weight(score_ksh_flattened)  # (B*T, n_e)

        # choose out prototype prob for each soft key (after update...)
        w_max_flattened = torch.gather(w_flattened, dim=-1, index=encoding_indices_flattened.unsqueeze(-1)).squeeze(-1)
        w_max = w_max_flattened.view(B, T)

        # calc score_kss
        score_kss = torch.einsum('btc,bcs->bts', key_soft_norm,
                                 rearrange(key_soft_norm, 'b t c -> b c t'))  # (B, T, T)

        # to weighted params
        vparams_norm = F.normalize(self.vparams.weight, p=2.0, dim=-1)  # (n_e, e_dim)
        vparams_w_flattened = w_flattened.view(B * T, 1, self.n_e) @ vparams_norm.view(1, self.n_e, self.e_dim).detach()
        vparams_w_flattened = vparams_w_flattened.squeeze(1)  # (B*T, e_dim)
        vparams_w_flattened = F.normalize(vparams_w_flattened, p=2.0, dim=-1)  # (B*T, e_dim)
        vparams_w = vparams_w_flattened.view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_w = vparams_w.contiguous()  # (B, T, e_dim)

        # calc score_vpsh (cos sim between vparams_w and self.vparams)
        score_vpsh_flattened = torch.einsum('bd,dn->bn', vparams_w_flattened,
                                            rearrange(vparams_norm, 'n d -> d n'))  # (B*T, n_e)
        score_vpsh = score_vpsh_flattened.view(B, T, self.n_e)  # (B, T, n_e)

        # calc score_vpss (cos sim between vparams_w)
        score_vpss = torch.einsum('btc,bcs->bts', vparams_w,
                                  rearrange(vparams_w, 'b t c -> b c t'))  # (B, T, T)

        vparams_hard = self.vparams(encoding_indices).view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_hard = F.normalize(vparams_hard, p=2.0, dim=-1)  # normalization (B, T, e_dim)
        vparams_hard = vparams_hard.contiguous()
        vparams_hard = vparams_w + (vparams_hard - vparams_w).detach()  # store gradient

        v_global = torch.max(encoding_indices) - torch.min(encoding_indices)

        return \
            encoding_indices, v_global, \
            vparams_w, vparams_hard, w_max, \
            score_vpss, score_vpsh, score_kss, score_ksh

    def forward(self, key_soft):
        # key_soft shape (B, T, self.key_dim)
        if self.use_ema:
            return self.ema_forward(key_soft)

        print('should not reach here')

        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]

        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)  # (B, T, key_dim)
        key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)  # (B*T, key_dim)
        keys_norm = F.normalize(self.keys.weight, p=2.0, dim=-1)  # (n_e, key_dim)

        # calc score_ksh
        score_ksh_flattened = torch.einsum('bd,dn->bn', key_soft_flattened,
                                           rearrange(keys_norm, 'n d -> d n'))  # (B*T, n_e)
        score_ksh = score_ksh_flattened.view(B, T, self.n_e)  # (B, T, n_e)

        # calc score_kss
        score_kss = torch.einsum('btc,bcs->bts', key_soft_norm,
                                 rearrange(key_soft_norm, 'b t c -> b c t'))  # (B, T, T)

        # to weighted params
        w_flattened = self.score_to_weight(score_ksh_flattened)  # (B*T, n_e)
        vparams_norm = F.normalize(self.vparams.weight, p=2.0, dim=-1)  # (n_e, e_dim)
        vparams_w_flattened = w_flattened.view(B * T, 1, self.n_e) @ vparams_norm.view(1, self.n_e, self.e_dim).detach()
        vparams_w_flattened = vparams_w_flattened.squeeze(1)  # (B*T, e_dim)
        vparams_w_flattened = F.normalize(vparams_w_flattened, p=2.0, dim=-1)  # (B*T, e_dim)
        vparams_w = vparams_w_flattened.view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_w = vparams_w.contiguous()  # (B, T, e_dim)

        # calc score_vpsh (cos sim between vparams_w and self.vparams)
        score_vpsh_flattened = torch.einsum('bd,dn->bn', vparams_w_flattened,
                                            rearrange(vparams_norm, 'n d -> d n'))  # (B*T, n_e)
        score_vpsh = score_vpsh_flattened.view(B, T, self.n_e)  # (B, T, n_e)

        # calc score_vpss (cos sim between vparams_w)
        score_vpss = torch.einsum('btc,bcs->bts', vparams_w,
                                  rearrange(vparams_w, 'b t c -> b c t'))  # (B, T, T)

        # Neareset Neighbor for indices and hard vparams
        encoding_indices = torch.argmax(score_ksh_flattened, dim=1)  # (B*T)

        w_max_flattened = torch.max(w_flattened, dim=-1)[0]
        w_max = w_max_flattened.view(B, T)
        encoding_indices = encoding_indices.view(B, T)  # (B, T)

        vparams_hard = self.vparams(encoding_indices).view(B, T, self.e_dim)  # (B, T, e_dim)
        vparams_hard = F.normalize(vparams_hard, p=2.0, dim=-1)  # normalization (B, T, e_dim)
        vparams_hard = vparams_hard.contiguous()
        vparams_hard = vparams_w + (vparams_hard - vparams_w).detach()  # store gradient

        v_global = torch.max(encoding_indices) - torch.min(encoding_indices)

        return \
            encoding_indices, v_global, \
            vparams_w, vparams_hard, w_max, \
            score_vpss, score_vpsh, score_kss, score_ksh

class VQClassifier(nn.Module):
    def __init__(self, key_dim, n_e, e_dim, KT=1.0):
        super().__init__()
        self.key_dim = key_dim
        self.n_e = n_e
        self.e_dim = e_dim
        self.KT = KT

        self.sm = nn.Softmax(dim=-1)

        self.classifier_linear = nn.Embedding(n_e, key_dim)
        self.classifier_linear.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        # mask for clustering between key_soft and key_hard
        # arange = torch.arange(n_e, dtype=torch.float32)
        # hm_dis = torch.abs(arange.view(-1, 1) - arange)
        # hm_dis = torch.neg(hm_dis - torch.ones_like(hm_dis))
        # hm_dis[[i for i in range(n_e)], [i for i in range(n_e)]] = float('-inf')
        # mask_ksh = torch.pow(2.0, hm_dis)
        # print('mask_ksh =\n', mask_ksh)
        # self.register_buffer('mask_ksh', mask_ksh)

        arange = torch.arange(n_e, dtype=torch.float32)
        hm_dis = torch.abs(arange.view(-1, 1) - arange)
        mask_khh = torch.where(torch.less_equal(hm_dis, 1), 1.0, 0.0)
        print('mask_khh =\n', mask_khh)
        self.register_buffer('mask_khh', mask_khh)

        # other helpful buffer
        self.register_buffer('arange', arange)

    # positive sample mask for (ks, ks) and (ks, kh)
    # indices: (B, T)
    # return, pn_ss (B, T, T), pn_sh (B, T, n_e), weighted (+1 or -1)
    def get_pn_mask(self, indices):
        # first get positive_sh
        B, T = indices.shape
        arange_sh = torch.less_equal(abs(indices.view(B, T, 1).repeat(1, 1, self.n_e)
                                         - self.arange.view(1, 1, self.n_e).repeat(B, T, 1)), 1)
        pn_sh = torch.where(arange_sh, 1.0, -1.0)

        arange_ss = torch.less_equal(abs(indices.view(B, 1, T).repeat(1, T, 1)
                                         - indices.view(B, T, 1).repeat(1, 1, T)), 1)
        pn_ss = torch.where(arange_ss, 1.0, -1.0)
        return pn_sh, pn_ss

    def score_to_weight(self, score):
        # score (B, T, n_e)
        weight = self.sm(torch.div(score, self.KT))  # (B, T, n_e)
        weight = torch.nan_to_num(weight)
        return weight

    def key_cos_sim(self, key_soft):
        # key_soft (B, T, C)
        with torch.no_grad():
            B = key_soft.shape[0]
            key_soft_norm = F.normalize(key_soft, dim=-1)

            key_hard = self.classifier_linear.weight
            key_hard_norm = F.normalize(key_hard, dim=-1)
            key_hard_norm = key_hard_norm.view(1, self.n_e, self.key_dim).repeat(B, 1, 1)

            k_cs_ss = torch.bmm(key_soft_norm, rearrange(key_soft_norm, 'b t c -> b c t'))
            k_cs_sh = torch.bmm(key_soft_norm, rearrange(key_hard_norm, 'b t c -> b c t'))

            return k_cs_ss, k_cs_sh

    # return: pn_change_ss, pn_change_sh
    def key_cs_change(self, k_cs1, k_cs2):
        k_cs_ss1, k_cs_sh1 = k_cs1
        k_cs_ss2, k_cs_sh2 = k_cs2
        with torch.no_grad():
            k_cs_ss_change = k_cs_ss2 - k_cs_ss1
            k_cs_sh_change = k_cs_sh2 - k_cs_sh1

            pn_change_ss = torch.where(torch.less(k_cs_ss_change, torch.zeros_like(k_cs_ss_change)),
                                       -1.0, 1.0)
            pn_change_sh = torch.where(torch.less(k_cs_sh_change, torch.zeros_like(k_cs_sh_change)),
                                       -1.0, 1.0)
            return pn_change_ss, pn_change_sh

    # def weight_to_loss(self, weight, indices):
    #     # weight (B, T, n_e)
    #     # indices (B, T)
    #     weight_i = torch.gather(
    #         input=weight,
    #         index=indices.unsqueeze(-1), dim=-1
    #     ).squeeze(-1)  # (B * T)
    #     log_weight_i = torch.log(weight_i)  # (B * T)
    #     log_weight_i = torch.nan_to_num(log_weight_i)  # (B * T)
    #     return torch.neg(log_weight_i)

    def get_key_soft_indices(self, key_soft, zero_ts_mask=None):
        # key_soft: (B, T, self.e_dim)
        # choosing indices does not need gradient
        with torch.no_grad():
            key_soft = key_soft.contiguous()
            B, T = key_soft.shape[0], key_soft.shape[1]
            key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
            key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)

            classifier_linear_norm = F.normalize(self.classifier_linear.weight, p=2.0, dim=-1)

            score_ksh = torch.einsum('bd,dn->bn', key_soft_flattened,
                                     rearrange(classifier_linear_norm, 'n d -> d n'))
            score_ksh = score_ksh.view(B, T, self.n_e)  # (B, T, n_e)

            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmax(score_ksh[:, 0, :], dim=1),
                min=0,
                max=(self.n_e-1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T - 1, 1):
                score_t = score_ksh[:, (t + 1), :]

                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=score_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=score_t, dim=1, index=ind_next)

                d_choice_mask = torch.greater_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

            return encoding_indices

    def policy_forward(self, key_soft, zero_ts_mask=None):
        # key_soft shape (B, T, self.key_dim)
        # zero_ts_mask (B)
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
        key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)
        classifier_linear_norm = F.normalize(self.classifier_linear.weight, p=2.0, dim=-1)
        score_ksh = torch.einsum('bd,dn->bn', key_soft_flattened,
                                 rearrange(classifier_linear_norm, 'n d -> d n'))
        score_ksh = score_ksh.view(B, T, self.n_e)

        # to key hard (weighed book)
        weight = self.score_to_weight(score_ksh)  # (B, T, n_e)
        key_hard = weight.view(B, T, 1, self.n_e) @ self.embedding.weight.view(1, 1, self.n_e, self.e_dim)
        key_hard = key_hard.squeeze(2)

        key_hard = key_hard.contiguous()

        with torch.no_grad():
            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmax(score_ksh[:, 0, :], dim=1),
                min=0,
                max=(self.n_e-1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T - 1, 1):
                score_t = score_ksh[:, (t + 1), :]

                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=score_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=score_t, dim=1, index=ind_next)

                d_choice_mask = torch.greater_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        key_hard_real = self.embedding(encoding_indices.view(-1)).view(B, T, self.e_dim)  # (B, T, e_dim)
        key_hard_real = key_hard_real.contiguous()

        # store gradient
        key_hard_real = key_hard + (key_hard_real - key_hard).detach()

        return key_hard_real

    def clustering_forward(self, key_soft, zero_ts_mask=None):
        # key_soft shape (B, T, key_dim)
        # zero_ts_mask (B)

        # clustering forward process will not pass anything to policy & energy part!
        key_soft = key_soft.contiguous()
        B, T = key_soft.shape[0], key_soft.shape[1]
        key_soft_norm = F.normalize(key_soft, p=2.0, dim=-1)
        key_soft_flattened = key_soft_norm.view(B * T, self.key_dim)
        classifier_linear_norm = F.normalize(self.classifier_linear.weight, p=2.0, dim=-1)
        score_ksh = torch.einsum('bd,dn->bn', key_soft_flattened,
                                 rearrange(classifier_linear_norm, 'n d -> d n'))
        score_ksh = score_ksh.view(B, T, self.n_e)
        score_ksh = torch.div(score_ksh, self.KT)

        score_kss = torch.einsum('btc,bcs->bts', key_soft_norm,
                                 rearrange(key_soft_norm, 'b t c -> b c t'))
        score_kss = torch.div(score_kss, self.KT)

        # score_khs = torch.einsum('bd,dn->bn', key_soft_flattened.detach(),
        #                          rearrange(self.classifier_linear.weight, 'n d -> d n'))
        # score_khs = score_khs.view(B, T, self.n_e)  # (B, T, n_e)

        # score_khh = torch.einsum('nd,dm->nm', self.classifier_linear.weight,
        #                          rearrange(self.classifier_linear.weight, 'n d -> d n'))

        # choosing indices does not need gradient
        with torch.no_grad():
            # First timestep (0 step) in every context will use the nearest key
            encoding_indices = torch.clip(
                torch.argmax(score_ksh[:, 0, :], dim=1),
                min=0,
                max=(self.n_e-1)
            )
            if zero_ts_mask is not None:
                encoding_indices = torch.where(zero_ts_mask, 0, encoding_indices)
            encoding_indices = encoding_indices.unsqueeze(1)

            # for the rest timestep, they will only select between form k_hard and its succesive key k_hard+
            # e.g. when your hard key is #1 at context timestep 0,
            #      your hard key is from {#1, #2} at context timestep 1,
            for t in range(0, T - 1, 1):
                score_t = score_ksh[:, (t + 1), :]

                ind_here = encoding_indices[:, t:(t + 1)]
                d_here = torch.gather(input=score_t, dim=1, index=ind_here)
                ind_next = torch.clip(ind_here + 1, min=0, max=(self.n_e - 1))
                d_next = torch.gather(input=score_t, dim=1, index=ind_next)

                d_choice_mask = torch.greater_equal(d_here, d_next)
                ind_new = torch.where(d_choice_mask, ind_here, ind_next)
                encoding_indices = torch.cat([encoding_indices, ind_new], dim=1)

        # # generate mask for score_khs, score_ksh
        # mask_score_ksh = self.mask_ksh[encoding_indices]

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)
        v_global = torch.max(encoding_indices) - torch.min(encoding_indices)

        return \
            encoding_indices, v, v_global, score_kss, score_ksh

    def forward(self, key_soft, zero_ts_mask=None, mode='policy'):
        # key_soft shape (B, T, self.key_dim)
        # zero_ts_mask (B)
        if mode == 'policy':
            return self.policy_forward(key_soft)
        elif mode == 'cluster':
            return self.clustering_forward(key_soft, zero_ts_mask)
        else:
            assert False


class VQNeighborBasic(nn.Module):
    def __init__(self, n_e, e_dim, legacy_cluster=0.2):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.legacy_cluster = legacy_cluster
        self.embedding = nn.Embedding(1 + n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        # self.register_buffer('arranged_mask', torch.arange(n_e + 1)[:, None])

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
            torch.sum((key_soft.detach() - key_hard_here) ** 2, dim=-1) * self.legacy_cluster \
            + torch.sum((key_soft - key_hard_here.detach()) ** 2, dim=-1)  # * self.legacy_cluster
        loss_next_base = \
            torch.sum((key_soft.detach() - key_hard_next) ** 2, dim=-1) * self.legacy_cluster \
            + torch.sum((key_soft - key_hard_next.detach()) ** 2, dim=-1)  # * self.legacy_cluster
        loss_min_indices = \
            torch.sum((key_soft.detach() - key_min) ** 2, dim=-1) \
            + torch.sum((key_soft - key_min.detach()) ** 2, dim=-1) * self.legacy_cluster
        # print(loss_min_indices.shape)
        loss_min_here = torch.where(torch.less(loss_min_indices, loss_here_base), loss_min_indices, 0.0)
        loss_min_next = torch.where(torch.less(loss_min_indices, loss_next_base), loss_min_indices, 0.0)
        loss_here = loss_here_base - loss_min_here
        loss_next = loss_next_base - loss_min_next

        # preserve gradients
        if preserve_grad:
            key_hard = key_soft + (key_hard_here - key_soft).detach()
        key_hard = key_hard.contiguous()

        min_indices, _ = torch.min(encoding_indices, dim=1)
        max_indices, _ = torch.max(encoding_indices, dim=1)
        v = torch.max(max_indices - min_indices)

        return \
            key_hard, encoding_indices, v, \
            loss_here, loss_next


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
        loss_next_reg = -loss_here_base
        loss_here_reg = -loss_next_base

        # energy
        # q_soft = self.key_to_qe(key_soft)
        # q_hard_here = self.key_to_qe(key_hard_here)
        # q_hard_next = self.key_to_qe(key_hard_next)

        q_soft = key_soft
        q_hard_here = key_hard_here
        q_hard_next = key_hard_next

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
