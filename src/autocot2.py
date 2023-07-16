import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.VQ import VQ2
from module.GPT2 import BasicNet, ActNet, ExampleNet

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
    masks = (temp < lengths.expand(B, max_len)).float()  # B x max_len

    loss = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss


def diff_cos_sim_loss(v, bound=None):
    diff_v = v[:, 1:, ...] - v[:, :-1, ...]
    diff_v_norm = F.normalize(diff_v, p=2, dim=2)
    diff_v_cos_sim = F.cosine_similarity(diff_v_norm[:, :-1, ...], diff_v_norm[:, 1:, ...], dim=2)
    if bound is not None:
        diff_v_cos_sim = torch.where(diff_v_cos_sim >= bound, bound, diff_v_cos_sim)
    loss = diff_v_cos_sim.mean()
    return loss


def begin_cos_sim_loss(v, bound=None):
    v_begin = v[:, 1:, ...] - v[:, 0:1, ...]
    v_begin_norm = F.normalize(v_begin, p=2, dim=2)
    v_begin_cos_sim = F.cosine_similarity(v_begin_norm[:, :-1, ...], v_begin_norm[:, -1:, ...], dim=2)
    if bound is not None:
        v_begin_cos_sim = torch.where(v_begin_cos_sim >= bound, bound, v_begin_cos_sim)
    loss = v_begin_cos_sim.mean()
    return loss


def key_states_movement_loss(key_hard_plus, key_commit, key_commit_next, marginal):
    return F.triplet_margin_loss(
        anchor=key_hard_plus,
        positive=key_commit_next,
        negative=key_commit,
        margin=marginal,
        p=2.0
    )



def init_centriods(datas, n_centriods):
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    cent_init = datas[i0][None, :]
    for _ in range(n_centriods - 1):
        d = torch.sum(datas ** 2, dim=1, keepdim=True) + \
            torch.sum(cent_init ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', datas, rearrange(cent_init, 'n d -> d n'))

        d_min = d.min(dim=1)[0] + 1e-5
        d_min = torch.where(torch.isnan(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(d_min < 0.0, torch.zeros_like(d_min), d_min)
        d_min_sum = d_min.sum()

        if d_min_sum == 0.0 or torch.isnan(d_min_sum) or torch.isinf(d_min_sum):
            i = torch.randint(0, N, (1,))[0]
        else:
            i = d_min.multinomial(num_samples=1)[0]

        cent_init = torch.cat([cent_init, datas[i][None, :]], dim=0)
    return cent_init


class RootConfig:
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, block_type, n_layer,
                 do_commit, max_timestep):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.block_size = block_size
        self.block_type = block_type
        self.n_layer = n_layer
        self.do_commit = do_commit
        self.max_timestep = max_timestep


class BasicNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer,
                 do_commit, sub_pos, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'basic', n_layer,
            do_commit, max_timestep
        )
        self.sub_pos = sub_pos


class ActNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, seq_k,
                 do_commit, sub_pos, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'act', n_layer,
            do_commit, max_timestep
        )
        self.seq_k = seq_k
        self.sub_pos = sub_pos


class ExampleNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'basic', n_layer,
            do_commit=False, max_timestep=max_timestep
        )


class AutoCoT(pl.LightningModule):
    def __init__(
            self,
            key_config,
            vq_n_e,
            vq_beta,
            vq_legacy,
            act_config,
            commit_config=None,
            coe_example=0.0,
            example_config=None,
            vq_smooth=None,
            vq_log=True,
            vq_kmeans_reset=None,
            vq_kmeans_step=None,
            use_soft_commit_loss=False,
            optimizers_config=None,
            scheduler_config=None,
            coe_reg_diff_k=None,
            bound_reg_diff_k=None,
            coe_reg_begin_k=None,
            bound_reg_begin_k=None,
            state_dim=-1,
            action_dim=-1
    ):
        super().__init__()

        self.use_soft_commit_loss = use_soft_commit_loss
        self.optimizers_config = optimizers_config
        self.scheduler_config = scheduler_config

        # choose to do commitment at KeyNet, ActNet, or an indenpendent CommitNet!
        use_keynet_commit = key_config.do_commit
        use_actnet_commit = act_config.do_commit
        use_commitnet = commit_config is not None
        assert (use_keynet_commit and not use_actnet_commit and not use_commitnet) \
               or (not use_keynet_commit and use_actnet_commit and not use_commitnet) \
               or (not use_keynet_commit and not use_actnet_commit and use_commitnet) \
               or (not use_keynet_commit and not use_actnet_commit and not use_commitnet)

        # When you use a indenpendent net to do commit, make sure do_commit is True
        if use_commitnet:
            assert commit_config.do_commit is True

        assert state_dim > 0 and action_dim > 0
        self.n_embd = key_config.n_embd

        # key_net, use for latent key predict
        # if key_config.do_commit is True, we will predict key_commit with key_net as well
        self.key_net = BasicNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # key_book, use for vq mapping
        # every soft key will be mapped to a hard key in the book using nearest neighbor
        # assert e_dim is always the key_config.n_embd size
        self.key_book = VQ2(
            n_e=vq_n_e,
            e_dim=key_config.n_embd,
            beta=vq_beta,
            legacy=vq_legacy,
            log_choice=vq_log
        )
        self.vq_smooth = vq_smooth
        if vq_kmeans_reset is not None:
            assert isinstance(vq_kmeans_reset, int)
            assert isinstance(vq_kmeans_step, int)
            self.kmeans_idx = 0
            self.vq_kmeans_reset = vq_kmeans_reset
            self.vq_kmeans_step = vq_kmeans_step
        else:
            self.kmeans_idx = None
            self.vq_kmeans_reset = None
            self.vq_kmeans_step = None

        # act_net, use for action prediction
        # if act_config.do_commit is True, we will predict key_commit with act_net as well
        self.act_net = ActNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # record commitment method:
        # 'independent': use an independent BasicNet
        # 'act': along with act_net
        # 'key': along with key_net
        if use_commitnet:
            self.commit_net = BasicNet(
                config=commit_config,
                state_dim=state_dim,
                action_dim=action_dim
            )
            self.commit_type = 'independent'
        elif use_actnet_commit:
            self.commit_type = 'act'
        elif use_keynet_commit:
            self.commit_type = 'key'
        else:
            self.commit_type = 'none'
        assert self.use_soft_commit_loss is False

        self.coe_example = coe_example

        if coe_example > 0.0:
            assert example_config is not None
            if coe_example >= 1.0:
                assert self.commit_type == 'none'
            # use example_net to reconstruct k[0:t-1] to s[0:t-1], a[0:t-2]
            self.example_net = ExampleNet(
                config=example_config,
                state_dim=state_dim,
                action_dim=action_dim
            )

        self.coe_reg_diff_k = coe_reg_diff_k
        self.bound_reg_diff_k = bound_reg_diff_k
        self.coe_reg_begin_k = coe_reg_begin_k
        self.bound_reg_begin_k = bound_reg_begin_k

        # flag_collapse: use for record whether we are in 'index collapse' situation
        self.flag_collapse = False
        self.kmeans_idx = 0

    @torch.no_grad()
    def kmeans_reset(self, key_soft):
        print('kmeans resetting...')
        arranged_mask = torch.arange(self.key_book.n_e)[:, None]
        arranged_mask = arranged_mask.to(self.device)

        key_soft_flatten = key_soft.view(-1, self.n_embd)

        # kmeans++ initialization
        self.key_book.embedding.weight.data = init_centriods(key_soft_flatten, self.key_book.n_e)

        # Do kmeans algorithm to reset
        for _ in range(self.vq_kmeans_step):
            _, _, indices, _ = self.key_book(key_soft_flatten, flatten_in=True, flatten_out=True)
            expanded_indices = indices[None].expand(self.key_book.n_e, -1)
            mask = (expanded_indices == arranged_mask).to(key_soft_flatten.dtype)
            c_grad = mask @ key_soft_flatten / mask.sum(-1)[..., :, None]
            torch.nan_to_num_(c_grad)
            self.key_book.embedding.weight.data = c_grad
        print('kmeans resetting done')

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.flag_collapse or ((self.vq_kmeans_reset is not None) and (self.kmeans_idx % self.vq_kmeans_reset == 0)):
            states, timesteps, actions, _ = batch['s'], batch['t'], batch['a'], batch['lengths']
            key_soft, _ = self.key_net(states, timesteps, actions)
            self.kmeans_reset(key_soft)
        self.kmeans_idx += 1

    def training_step(self, batch, batch_idx):
        states, timesteps, actions, lengths = batch['s'], batch['t'], batch['a'], batch['lengths']
        T = states.shape[1]

        # Latent encoding
        # from s[0:T-1], a[0:T-1] get key_soft[0:T-1]
        # if commit, from s[0:T-1], a[0:T-2] get kcs_key_net[0:T-1]
        key_soft, kcs_key_net = self.key_net(states, timesteps, actions)

        reg_k_line = 0.0
        if self.coe_reg_diff_k is not None:
            reg_diff_k_line = diff_cos_sim_loss(key_soft, self.bound_reg_diff_k)
            reg_k_line = reg_k_line + self.coe_reg_diff_k * reg_diff_k_line
        else:
            reg_diff_k_line = 0.0
        if self.coe_reg_begin_k is not None:
            reg_begin_k_line = begin_cos_sim_loss(key_soft, self.bound_reg_begin_k)
            reg_k_line = reg_k_line + self.coe_reg_begin_k * reg_begin_k_line
        else:
            reg_begin_k_line = 0.0

        # VQ mapping
        key_hard, loss_dict, indices, var = self.key_book(key_soft)
        # print('VQ mapping, indices shape', indices.shape)

        if var == 0 and self.flag_collapse is False:
            self.flag_collapse = True
        elif var != 0 and self.flag_collapse is True:
            self.flag_collapse = False


        if self.coe_example > 0.0:
            # reconstruction from key_hard to s[0:t-1], a[0:t-2]
            if self.vq_smooth is not None:
                key_smooth = (1.0 - self.vq_smooth) * key_hard + self.vq_smooth * key_soft
            else:
                key_smooth = key_hard
            state_recs, act_recs = self.example_net(key_smooth, timesteps)
            loss_recs_state = get_loss(state_recs, states, lengths)
            if act_recs is not None:
                loss_recs_action = get_loss(act_recs, actions[:, 0:(T-1)], lengths - 1)
            else:
                loss_recs_action = 0.0
            loss_recs = loss_recs_state + loss_recs_action
        else:
            loss_recs = 0.0

        # action based on k_hard[0:T-1]
        # from s[0:T-1], a[0:T-2], k_hard[0:T-1] get a[0:T-1]
        # if commit, from s[0:T-1], a[0:T-2] get k[0:T-1]
        act_preds, kcs_act_net = self.act_net(states, timesteps, actions, key_hard)

        # commitment
        if self.commit_type == 'none':
            # do not train commit_net
            key_commit_soft = None
        elif self.commit_type == 'independent':
            # commit with commit_net
            _, key_commit_soft = self.commit_net(states, timesteps, actions)
        elif self.commit_type == 'key':
            # using act_net commit_ket
            key_commit_soft = kcs_key_net
        elif self.commit_type == 'act':
            # using act_net commit_ket
            key_commit_soft = kcs_act_net
        else:
            print('should not reach here')
            assert False

        # loss: prediction
        loss_preds = get_loss(act_preds, actions, lengths)
        # loss: commitment
        if key_commit_soft is not None:
            loss_commitment = torch.mean((key_commit_soft - key_hard.detach()) ** 2) + \
                              self.key_book.beta * torch.mean((key_commit_soft.detach() - key_hard) ** 2)
            # print(indices.shape)
            loss_key_movement = key_states_movement_loss(
                key_hard_plus=self.key_book.index_select(indices=indices + 1)[:, :-1, ...],   ### WRONG, NEED TO BE THE NEXT IN INDICE!!!
                key_commit=key_soft[:, :-1, ...],
                key_commit_next=key_soft[:, 1:, ...],
                marginal=1e-6,
            )

        else:
            loss_commitment = 0.0
            loss_key_movement = 0.0

        loss = \
            loss_preds \
            + self.coe_example * loss_recs \
            + (1.0 - self.coe_example) * (loss_commitment + loss_key_movement)\
            + loss_dict \
            - reg_k_line


        # log the loss-es
        self.log_dict(
            {
                'loss_preds': loss_preds,
                # 'loss_recs': loss_recs,
                'loss_commitment': loss_commitment,
                'loss_key_movement': loss_key_movement,
                'loss_dict': loss_dict,
                # 'reg_diff_k_line': reg_diff_k_line,
                # 'reg_begin_k_line': reg_begin_k_line
            }, prog_bar=True, on_step=True, on_epoch=True
        )


        # loss key_book choice variation
        if var is not None:
            self.log('choice_var', var, prog_bar=True, on_step=True, on_epoch=True)

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
        key_soft, _ = self.key_net(states, timesteps, actions)
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
        no_decay.add('key_net.local_pos_emb')
        no_decay.add('key_net.global_pos_emb')
        no_decay.add('act_net.local_pos_emb')
        no_decay.add('act_net.global_pos_emb')
        if self.commit_type == 'independent':
            no_decay.add('commit_net.local_pos_emb')
            no_decay.add('commit_net.global_pos_emb')
        if self.coe_example > 0.0:
            no_decay.add('example_net.local_pos_emb')
            no_decay.add('example_net.global_pos_emb')

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