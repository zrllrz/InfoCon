import numpy as np
from math import exp, pow
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.VQ import VQNeighbor, VQNeighbor2, VQElastic
from module.GPT import KeyNet, ActCommitNet, ENet, MLP

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from lr_scheduler import CosineAnnealingLRWarmup

from einops import rearrange

from util import anomaly_score, cos_anomaly_score, mse_loss_with_weights, get_loss, init_centroids, init_centroids_neighbor


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


class ENetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, '-', n_layer, max_timestep
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
                 block_size, n_layer, max_timestep,
                 commit, use_key_energy):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.commit = commit
        self.use_key_energy = use_key_energy


class AutoCoT(pl.LightningModule):
    def __init__(
        self,
        key_config,
        act_config,
        e_config,
        vq_n_e=10,
        vq_legacy_cluster=0.2,
        optimizers_config=None,
        scheduler_config=None,
        state_dim=-1,
        action_dim=-1,
        key_dim=-1
    ):
        super().__init__()

        self.optimizers_config = optimizers_config
        self.scheduler_config = scheduler_config

        assert state_dim > 0 and action_dim > 0
        self.n_embd = key_config.n_embd

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

        self.key_book = VQNeighbor2(
            n_e=vq_n_e,
            e_dim=key_dim,
            legacy_cluster=vq_legacy_cluster,
        )

        # act_net, for action prediction
        assert act_config.commit is False
        print(act_config.use_key_energy)
        self.act_net = ActCommitNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim,
            key_dim=key_dim
        )
        self.e_net = ENet(
            config=e_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def training_step(self, batch, batch_idx):
        states, timesteps, actions, unified_t, lengths = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
        states.requires_grad = True
        zero_ts_mask = torch.eq(unified_t[:, 0], 0)
        B, T = states.shape[0], states.shape[1]

        # Latent encoding
        # from s[0:T-1], a[0:T-1] get key_soft[0:T-1]
        key_soft, state_recs = self.key_net(states, timesteps, actions)
        # VQ mapping
        key_hard, encoding_indices, v, loss_here, loss_next, energy, loss_energy_descent = \
            self.key_book(key_soft, zero_ts_mask=zero_ts_mask)

        energy_grad = torch.autograd.grad(energy, states, create_graph=True, retain_graph=True)[0]

        # action prediction based on k_hard[0:T-1]
        # from s[0:T-1], a[0:T-2], k_hard[0:T-1] get a[0:T-1]
        if self.act_net.use_key_energy:
            act_preds = self.act_net(states, timesteps, actions, key_soft, key_energy=energy_grad)
        else:
            act_preds = self.act_net(states, timesteps, actions, key_soft, key_energy=None)

        # Try to increase mutual information between action and ∂E/∂s
        eact_preds = self.e_net(energy_grad, timesteps)

        # loss: state reconstruction
        l_s_recs, _ = get_loss(state_recs, states[:, 1:, ...], lengths - 1)

        # loss: action prediction & naive action prediction
        l_a_preds, ls_a_preds = get_loss(act_preds, actions, lengths)
        l_ea_preds, ls_ea_preds = get_loss(eact_preds, actions, lengths)
        l_preds = l_a_preds + l_ea_preds

        # first use ∂(l_a_preds)/∂(key_soft) and ∂(l_ea_preds)/∂(key_soft) to
        # judge the moving direction of key_soft, you can rely on the direction to decide whether to
        # move key_soft closer to key_hard_here and key_hard_next
        key_soft_md = torch.autograd.grad(l_preds, key_soft, create_graph=True, retain_graph=True)[0]
        key_soft_md = key_soft_md.detach()  # (B, T, key_dim)
        key_soft_to_here = key_hard - key_soft  # (B, T, key_dim)
        key_soft_to_here = key_soft_to_here.detach()

        # larger cos means the tendency of updating key_soft will lead to closer to your current cluster
        # otherwise key_soft is tend to move farther from the current key_hard(_here)
        # (you can comply the tendency in this situation!!!)
        key_d_cos = F.cosine_similarity(key_soft_md, key_soft_to_here, dim=-1)

        # a strange activation unit, when cos >= 0.0, normal (agree with the current key_hard(_here) choice)
        # when cos < 0.0, anomaly, try to move the k_s towards the key_hard_next
        w_anomaly = cos_anomaly_score(key_d_cos, sharpen=10.0)
        # print('w_anomaly.shape:', w_anomaly.shape)

        # print(losses_preds.shape)
        # loss_range = self.key_book.get_loss_structure()

        # clustering loss

        loss = \
            l_preds + l_s_recs \
            + ((1.0 - w_anomaly) * loss_here).mean() + (w_anomaly * loss_next).mean() \
            + loss_energy_descent  # + loss_range * self.key_book.coe_structure

        # log the loss-es
        self.log_dict(
            {
                'v': v,
                '?s': l_s_recs,
                '?a': l_a_preds,
                '?ea': l_ea_preds,
                '∂E': loss_energy_descent,
                '©': loss_here.mean(),
                'mab': torch.max(w_anomaly),
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
        key_soft, _ = self.key_net(states, timesteps, actions)
        label, _, _ = self.key_book.get_key_soft_indices(key_soft, zero_ts_mask=None)
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
        # no_decay.add('rec_net.local_pos_emb')
        # no_decay.add('rec_net.global_pos_emb')
        no_decay.add('key_net.local_pos_emb')
        no_decay.add('key_net.global_pos_emb')
        no_decay.add('act_net.local_pos_emb')
        no_decay.add('act_net.global_pos_emb')
        no_decay.add('e_net.local_pos_emb')
        no_decay.add('e_net.global_pos_emb')
        # no_decay.add('commit_net.local_pos_emb')
        # no_decay.add('commit_net.global_pos_emb')


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