import numpy as np
from math import exp, pow
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.VQ import VQNeighbor
from module.GPT import KeyNet, ActCommitNet, RecNet

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from lr_scheduler import CosineAnnealingLRWarmup

from einops import rearrange


def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape, losses.shape
        return torch.mean(losses * weights)


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

    loss = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss


def key_states_movement_loss(key_hard_plus, key_commit, key_commit_next, marginal):
    return F.triplet_margin_loss(
        anchor=key_hard_plus,
        positive=key_commit_next,
        negative=key_commit,
        margin=marginal,
        p=2.0
    )


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


class RootConfig:
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, attn_type, n_layer, max_timestep):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.block_size = block_size
        self.attn_type = attn_type
        self.n_layer = n_layer
        self.max_timestep = max_timestep


class KeyNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'wo_key', n_layer, max_timestep
        )


class RecNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, '-', n_layer, max_timestep
        )


class ActCommitNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep, commit):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.commit = commit


class AutoCoT(pl.LightningModule):
    def __init__(
        self,
        rec_config,
        key_config,
        vq_n_e,
        vq_beta,
        vq_legacy,
        act_config,
        commit_config,
        vq_log=True,
        vq_persistence=None,
        vq_kmeans_reset=None,
        vq_kmeans_step=None,
        optimizers_config=None,
        scheduler_config=None,
        subgoal_marginal=1e-6,
        state_dim=-1,
        action_dim=-1,
        key_dim=-1
    ):
        super().__init__()

        self.optimizers_config = optimizers_config
        self.scheduler_config = scheduler_config

        assert state_dim > 0 and action_dim > 0
        assert rec_config.n_layer == key_config.n_layer
        self.n_embd = key_config.n_embd

        self.rec_net = RecNet(
            config=rec_config,
            state_dim=state_dim,
            key_dim=key_dim
        )

        # key_net, use for latent key recognition
        self.key_net = KeyNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim,
            key_dim=key_dim
        )

        # key_book, use for vq mapping
        # every soft key will be mapped to a hard key in the book using a restrict nearest neighbour
        # which will make sure the indices are close...

        if vq_persistence is not None:
            persistence = vq_persistence / float(self.key_net.block_size)
        else:
            persistence = 1.0
        print(persistence)

        self.key_book = VQNeighbor(
            n_e=vq_n_e,
            e_dim=key_dim,
            beta=vq_beta,
            legacy=vq_legacy,
            log_choice=vq_log,
            persistence=persistence
        )
        if vq_kmeans_reset is not None:
            assert isinstance(vq_kmeans_reset, int)
            assert isinstance(vq_kmeans_step, int)
            self.flag_collapse = False
            self.kmeans_idx = 0
            self.vq_kmeans_reset = vq_kmeans_reset
            self.vq_kmeans_step = vq_kmeans_step
        else:
            self.kmeans_idx = None
            self.vq_kmeans_reset = None
            self.vq_kmeans_step = None

        # act_net, for action prediction
        assert act_config.commit is False
        self.act_net = ActCommitNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim,
            key_dim=key_dim
        )

        # commit_net, for committing key prediction
        assert commit_config.commit is True
        self.commit_net = ActCommitNet(
            config=commit_config,
            state_dim=state_dim,
            action_dim=action_dim,
            key_dim=key_dim
        )

        self.goalLoss = nn.TripletMarginLoss(
            margin=subgoal_marginal,
            p=2.0,
            reduction='mean'
        )

    @torch.no_grad()
    def kmeans_reset(self, key_soft, unified_t=None):
        print('kmeans resetting...')
        # arranged_mask = torch.arange(self.key_book.n_e + 1)[:, None]
        # arranged_mask = arranged_mask.to(self.device)
        arranged_mask = self.key_book.arranged_mask
        key_soft_flatten = key_soft.view(-1, self.n_embd)

        # kmeans++ initialization
        if unified_t is not None:
            self.key_book.embedding.weight.data = \
                init_centroids_neighbor(key_soft_flatten, unified_t.view(-1), self.key_book.n_e + 1)
        else:
            self.key_book.embedding.weight.data = \
                init_centroids(key_soft_flatten, self.key_book.n_e + 1)

        # Do kmeans algorithm to reset
        for _ in range(self.vq_kmeans_step):
            _, _, indices, _ = self.key_book(key_soft)
            indices_flatten = indices.view(-1)
            expanded_indices = indices_flatten[None].expand(self.key_book.n_e + 1, -1)
            mask = torch.eq(expanded_indices, arranged_mask).to(key_soft_flatten.dtype)
            c_grad = mask @ key_soft_flatten / mask.sum(-1)[..., :, None]
            torch.nan_to_num_(c_grad)
            self.key_book.embedding.weight.data = c_grad
        print('kmeans resetting done')

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.vq_kmeans_reset is None:
            return
        else:
            if self.flag_collapse or (self.kmeans_idx % self.vq_kmeans_reset == 0):
                states, timesteps, actions, unified_t, _ = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
                key_soft, _ = self.key_net(states, timesteps, actions)
                self.kmeans_reset(key_soft, unified_t=unified_t)
            self.kmeans_idx += 1

    def training_step(self, batch, batch_idx):
        states, timesteps, actions, _, lengths = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
        T = states.shape[1]

        # Latent encoding
        # from s[0:T-1], a[0:T-1] get key_soft[0:T-1]
        key_soft, skip_features = self.key_net(states, timesteps, actions)

        state_recs = self.rec_net(key_soft, skip_features, timesteps)

        # VQ mapping
        key_hard, loss_dict, indices, var = self.key_book(key_soft)

        if self.vq_kmeans_reset is None:
            pass
        else:
            if var == 0 and self.flag_collapse is False:
                self.flag_collapse = True
            elif var != 0 and self.flag_collapse is True:
                self.flag_collapse = False

        # action prediction based on k_hard[0:T-1]
        # from s[0:T-1], a[0:T-2], k_hard[0:T-1] get a[0:T-1]
        act_preds = self.act_net(states, timesteps, actions, key_soft)

        # commitment
        # from s[0:T-1], k_hard[0:T-2], a[0:T-2] get k_hard[0:T-1]
        key_commit_soft = self.commit_net(states, timesteps, actions, key_soft)

        # loss: state reconstruction
        loss_recs = get_loss(state_recs, states, lengths)

        # loss: action prediction
        loss_preds = get_loss(act_preds, actions, lengths)

        # loss: commitment
        loss_commitment = \
            torch.mean((key_commit_soft - key_soft.detach()) ** 2) + \
            self.key_book.beta * torch.mean((key_commit_soft.detach() - key_soft) ** 2)

        # loss: subgoal
        # Your next key_soft should be more closer to the previous goal
        # k_hard(i-1)
        # k_soft(i-1)   k_soft(i)
        #             \     | (more closer!)
        #               k_hard(i-1)+
        #               (your goal at (i - 1), when you take action to transmit from k_hard(i-1) to k_hard(i-1)+)

        loss_subgoal = self.goalLoss(
            anchor=self.key_book.select_from_index(indices=indices[:, :-1] + 1).detach(),
            positive=key_soft[:, 1:, :],
            negative=key_soft[:, :-1, :].detach(),
        ) + self.key_book.beta * self.goalLoss(
            anchor=self.key_book.select_from_index(indices=indices[:, :-1] + 1),
            positive=key_soft[:, 1:, :].detach(),
            negative=key_soft[:, :-1, :],
        )
        #
        # loss_key_range = \
        #     torch.maximum(torch.abs(key_soft) - 1.0 / self.key_book.n_e, torch.zeros_like(key_soft)) + \
        #     torch.maximum(torch.abs(key_commit_soft) - 1.0 / self.key_book.n_e, torch.zeros_like(key_commit_soft))
        # loss_key_range = loss_key_range.sum()

        loss = loss_recs + loss_preds + loss_commitment + loss_subgoal + loss_dict  # + loss_key_range * 0.1

        # loss key_book choice variation
        if var is not None:
            self.log('choice_var', var, prog_bar=True, on_step=True, on_epoch=True)

        # log the loss-es
        self.log_dict(
            {
                'rec': loss_recs,
                'pre': loss_preds,
                'sub': loss_subgoal,
                # 'k_r': loss_key_range,
                'com': loss_commitment,
                'dic': loss_dict,
            }, prog_bar=True, on_step=True, on_epoch=True
        )

        return loss

    def label_single(self, states, timesteps, actions=None):
        # states: (T, s_dim)
        # actions: None or (T - 1, a_dim)
        # timesteps

        # states = states[None, ...]
        # timesteps = timesteps[None, ...]
        # if actions is not None:
        #     actions = actions[None, ...]

        # key_net label
        key_soft = self.key_net(states, timesteps, actions)
        _, _, label, _ = self.key_book(key_soft)
        return label[:, -1]

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the module into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # NEED MODIFICATION
        no_decay.add('rec_net.local_pos_emb')
        no_decay.add('rec_net.global_pos_emb')
        no_decay.add('key_net.local_pos_emb')
        no_decay.add('key_net.global_pos_emb')
        no_decay.add('act_net.local_pos_emb')
        no_decay.add('act_net.global_pos_emb')
        no_decay.add('commit_net.local_pos_emb')
        no_decay.add('commit_net.global_pos_emb')


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": self.optimizers_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        if self.optimizers_config is not None:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.optimizers_config['init_lr'],
                betas=(self.optimizers_config['beta1'], self.optimizers_config['beta2'])
            )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=1e-4,
                betas=(0.9, 0.95)
            )
        # scheduler config
        if self.scheduler_config is None:
            return optimizer

        assert 'type' in self.scheduler_config.keys()
        if self.scheduler_config['type'] == 'cos_decay_with_warmup':
            assert 't_max', 't_warmup' in self.scheduler_config.keys()
            scheduler = CosineAnnealingLRWarmup(
                optimizer,
                T_max=self.scheduler_config['t_max'],
                T_warmup=self.scheduler_config['t_warmup']
            )
        elif self.scheduler_config['type'] == 'multistep':
            assert 'milestones', 'gamma' in self.scheduler_config.keys()
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.scheduler_config['milestones'],
                gamma=self.scheduler_config['gamma']
            )

        return [optimizer], [scheduler]