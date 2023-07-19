import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.VQ import VQ3
from module.GPT3 import KeyCommitNet, ActNet

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


class RootConfig:
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, block_type, n_layer, max_timestep):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.block_size = block_size
        self.block_type = block_type
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
        self.do_commit = False


class CommitNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'wo_key', n_layer, max_timestep
        )
        self.do_commit = True


class ActNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )


class AutoCoK(pl.LightningModule):
    def __init__(
            self,
            key_config,
            vq_n_e,
            act_config,
            commit_config,
            vq_p_change_th=0.8,
            vq_log=True,
            optimizers_config=None,
            scheduler_config=None,
            state_dim=-1,
            action_dim=-1
    ):
        super().__init__()

        self.optimizers_config = optimizers_config
        self.scheduler_config = scheduler_config

        assert state_dim > 0 and action_dim > 0
        self.n_embd = key_config.n_embd

        # key_net, use for latent key predict
        # if key_config.do_commit is True, we will predict key_commit with key_net as well
        self.key_net = KeyCommitNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # key_book, use for vq mapping
        # every soft key will be mapped to a hard key in the book using nearest neighbor
        # assert e_dim is always the key_config.n_embd size
        self.key_book = VQ3(
            n_e=vq_n_e,
            e_dim=key_config.n_embd,
            p_change_th=vq_p_change_th,
            log_choice=vq_log
        )

        # act_net, use for action prediction
        # if act_config.do_commit is True, we will predict key_commit with act_net as well
        self.act_net = ActNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # record commitment method:
        self.commit_net = KeyCommitNet(
            config=commit_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def training_step(self, batch, batch_idx):
        states, timesteps, actions, lengths = batch['s'], batch['t'], batch['a'], batch['lengths']
        T = states.shape[1]

        # Latent encoding
        # from s[0:T-1], a[0:T-1] get key_soft[0:T-1]
        # if commit, from s[0:T-1], a[0:T-2] get kcs_key_net[0:T-1]
        minus_log_one_minus_p_change, p_change = self.key_net(states, timesteps, actions)

        # VQ mapping
        keys, var = self.key_book(p_change)

        # action based on k_hard[0:T-1]
        # from s[0:T-1], a[0:T-2], k_hard[0:T-1] get a[0:T-1]
        # if commit, from s[0:T-1], a[0:T-2] get k[0:T-1]
        act_preds = self.act_net(states, timesteps, actions, keys)

        # commitment
        minus_log_one_minus_p_change_commit, p_change_commit = \
            self.commit_net(states, timesteps, actions)

        # loss: prediction
        loss_preds = get_loss(
            preds=act_preds,
            targets=actions,
            lengths=lengths
        )
        # loss: commitment
        loss_commitment = F.binary_cross_entropy(
            input=p_change_commit,
            target=p_change,
        )
        # loss: subgoal
        loss_subgoal =

        loss = loss_preds + loss_commitment + loss_subgoal


        # log the loss-es
        self.log_dict(
            {
                'loss_preds': loss_preds,
                'loss_commitment': loss_commitment,
                'loss_subgoal': loss_subgoal,
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