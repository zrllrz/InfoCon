import numpy as np
from math import exp, pow, log
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.module_util import FreqEncoder, TimeSphereEncoder, mereNLL
from module.VQ import VQClassifierNN, VQClassifier, VQNeighbor, VQNeighbor2, VQElastic, VQNeighborBasic
from module.GPT import KeyNet, RecNet, ImplicitSAGPT, ExplicitSAGPT, ExplicitSAHNGPT, ActCommitNet, ENet, MLP
from module.ResNetFC import ImplicitSAResFC, ExplicitSAHN

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


class ImplicitSAResFCConfig:
    def __init__(self, n_embd, block_size, use_pos_emb, n_state_layer, n_action_layer, max_timestep):
        self.type = 'resfc'
        self.n_embd = n_embd
        self.n_state_layer = n_state_layer
        self.n_action_layer = n_action_layer
        self.block_size = block_size
        self.use_pos_emb = use_pos_emb
        self.max_timestep = max_timestep


class ExplicitSAHNConfig:
    def __init__(self, dim_h, block_size, use_pos_emb, reward_layer, max_timestep):
        self.type = 'hn'
        self.dim_h = dim_h
        self.reward_layer = reward_layer
        self.block_size = block_size
        self.use_pos_emb = use_pos_emb
        self.max_timestep = max_timestep


class ImplicitSAGPTConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, state_layer, max_timestep):
        self.type = 'gpt'
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.state_layer = state_layer


class ExplicitSAGPTConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, n_state_layer, max_timestep):
        self.type = 'egpt'
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.n_state_layer = n_state_layer

class ExplicitSAHNGPTConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, n_state_layer, use_skip, max_timestep):
        self.type = 'egpthn'
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.n_state_layer = n_state_layer
        self.use_skip = use_skip


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
        rec_config,
        sa_config,
        vq_n_e=10,
        vq_use_ema=True,
        vq_coe_ema=0.95,
        KT=2.0,
        optimizers_config=None,
        scheduler_config=None,
        state_dim=-1,
        action_dim=-1,
        key_dim=-1,
        e_dim=-1,
        use_st=True,
        rate_st=2.0,
        te_keys_dim=None,
        coe_lip_pos=1.0,
        coe_lip_neg=0.5,
        vq_fresh_step=1000,
    ):
        super().__init__()

        self.optimizers_config = optimizers_config
        self.scheduler_config = scheduler_config

        assert state_dim > 0 and action_dim > 0 and key_dim > 0 and e_dim > 0
        self.n_embd = key_config.n_embd

        # key_net, use for latent key recognition
        self.key_net = KeyNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim,
            key_dim=key_dim,
        )

        self.use_st = use_st
        if use_st:
            self.ks_t_emb = TimeSphereEncoder(rate=rate_st)
            tek_dim = 1
            self.vq_fresh_step = vq_fresh_step
        else:
            self.vq_fresh_step = None  # You Cannot use refresh and splitting key hards when using
            if te_keys_dim is not None:
                assert type(te_keys_dim) == int
                self.ks_t_emb = FreqEncoder(length=te_keys_dim)
                tek_dim = te_keys_dim * 2
            else:
                self.ks_t_emb = None
                tek_dim = 0

        self.rec_net = RecNet(
            config=rec_config,
            state_dim=state_dim,
            key_dim=key_dim+tek_dim
        )

        print('block size =', key_config.block_size)
        block_size = key_config.block_size

        # mask for key soft time-steps constraint
        check_mask_base = 1 - torch.tril(torch.ones(block_size, block_size))
        check_mask_sum = torch.cumsum(check_mask_base, dim=-1)
        check_mask_0 = ((check_mask_sum % 2 == 1) * check_mask_base).to(dtype=torch.float32)
        check_mask_1 = ((check_mask_sum % 2 == 0) * check_mask_base).to(dtype=torch.float32)
        check_mask_0 = check_mask_0[:-1, 1:]
        check_mask_1 = check_mask_1[:-1, 1:]
        print('check_mask_0 =\n', check_mask_0)
        print('check_mask_1 =\n', check_mask_1)
        self.register_buffer('cm', torch.stack([check_mask_0, check_mask_1], dim=0))

        # mask for clustering between key_soft
        arange = torch.arange(block_size, dtype=torch.float32)
        hm_dis = torch.abs(arange.view(-1, 1) - arange)
        hm_dis = torch.neg(hm_dis - torch.ones_like(hm_dis))
        hm_dis[[i for i in range(block_size)], [i for i in range(block_size)]] = float('-inf')
        w_ss = torch.pow(2.0, hm_dis * (vq_n_e - 2) / (block_size - 2))
        print('w_ss =\n', w_ss)
        self.register_buffer('w_ss', w_ss)

        # sa_net, for action, next_state prediction to eval the hard key
        self.sa_type = sa_config.type
        if sa_config.type == 'resfc':
            self.sa_net = ImplicitSAResFC(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=e_dim
            )
        elif sa_config.type == 'gpt':
            self.sa_net = ImplicitSAGPT(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=e_dim
            )
        elif sa_config.type == 'hn':
            self.sa_net = ExplicitSAHN(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                e_dim=e_dim * sa_config.reward_layer
            )
        elif sa_config.type == 'egpt':
            self.sa_net = ExplicitSAGPT(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=e_dim
            )
        elif sa_config.type == 'egpthn':
            self.sa_net = ExplicitSAHNGPT(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=e_dim
            )
        else:
            print('unknown sa_config.type')
            assert False

        # key_book, use for vq mapping
        # every soft key will be mapped to a hard key in the book using a restrict nearest neighbour
        # which will make sure the indices are close...
        self.key_book = VQClassifierNN(
            key_dim=key_dim+tek_dim,
            n_e=vq_n_e,
            e_dim=e_dim * sa_config.n_state_layer if (sa_config.type == 'hn' or sa_config.type == 'egpthn') else e_dim,
            KT=KT,
            use_ema=vq_use_ema,
            coe_ema=vq_coe_ema
        )

        # When training, we need to adjust the goal function part in sa_net
        # so that every goal function (params by the chosen out v_param) can discriminate
        # whether a certain state is at the finishing state of this situation
        # this is like we have vq_n_e x 2 class classifier
        # we do not choose to update all of them every iter
        # (it will be hard when vq_n_e is larger, while 'clustering itself is the source of collapse')
        # we will use a circle way to update them
        self.vq_n_e = vq_n_e
        self.vq_turn = 0  # start from 0, circle movement, update the first meet index
                          # which some states are classified to during this iter

        self.LIP_POS = coe_lip_pos
        self.LIP_NEG = coe_lip_neg

        if optimizers_config is not None:
            self.coe_cluster = optimizers_config['coe_cluster']
            self.coe_rec = optimizers_config['coe_rec']
        else:
            self.coe_cluster = 0.0
            self.coe_rec = 0.0

        self.step_refresh = 0
        self.ls_a_pred_mean = 0.0
        self.cnt_a_pred_mean = 0

        self.step_cluster = 0
        self.cyc_cluster = 1
        self.flag_cluster = False
        self.goal_cluster = True

        self.automatic_optimization = False

    def on_train_batch_start(self, batch, batch_idx):
        self.step_refresh += 1
        self.step_cluster += 1
        if self.step_cluster == self.cyc_cluster:
            self.step_cluster = 0
            self.flag_cluster = True
        else:
            self.flag_cluster = False

        if self.step_refresh == self.vq_fresh_step:
            # refresh key_book, do splitting
            self.split_keys(self.ls_a_pred_mean, tolerance=1.0)
            self.key_book.refresh_loss_label()
            self.step_refresh = 0
            self.ls_a_pred_mean = 0.0
            self.cnt_a_pred_mean = 0

    def split_keys(self, loss_mean, tolerance=1.0):
        with torch.no_grad():
            loss_label_max, loss_label_i = torch.max(self.key_book.loss_label, dim=-1)
            print('loss_label_max =', loss_label_max)
            if loss_label_max > loss_mean * (1.0 + tolerance):
                # try tp split loss_label_i
                # first find a (Latest Recent) unused (LRU) index
                ix = 0
                for ix in range(self.key_book.n_e):
                    if self.key_book.n_label[ix] == 0:
                        break
                loss_label_i_new = ix
                print('loss_label_i_new', loss_label_i_new)
                if loss_label_i_new >= self.key_book.n_e:
                    print('No Index Left. Give up Splitting')
                    return
                # else split the keys[loss_label_i], one in keys[loss_label_i], one in keys[loss_label_i_new]
                key_old = self.key_book.keys[loss_label_i]  # (key_dim)

                # assert at autocot that you are using time-step spherical embedding
                sin_t = key_old[0]
                key_old_wote = torch.div(key_old[1:], torch.sqrt(1.0 - sin_t ** 2) + 1e-9)
                # split into two new keys according to embedding of time
                key_new_1, key_new_2 = \
                    self.ks_t_emb(key_old_wote, (self.key_book.min_ut[loss_label_i] * 2.0 + self.key_book.max_ut[loss_label_i]) / 3.0), \
                    self.ks_t_emb(key_old_wote, (self.key_book.min_ut[loss_label_i] + self.key_book.max_ut[loss_label_i] * 2.0) / 3.0)

                # update the keys
                self.key_book.keys[loss_label_i] = key_new_1
                self.key_book.keys[loss_label_i_new] = key_new_2

    def statistic_indices(self, indices, unified_t=None):
        # (B, T)
        if unified_t is None:
            table = torch.zeros(size=(indices.shape[0], self.key_book.n_e))  # (B, n_e)
            for i in range(self.key_book.n_e):
                table[:, i] = torch.sum((indices == i).to(torch.int), dim=-1)
            print(table[:min(indices.shape[0], 32), ...])
        else:
            table = torch.zeros(size=(10, self.key_book.n_e), dtype=torch.int32)  # (B, n_e)
            for i in range(10):
                lbm = torch.ge(unified_t, i*0.1)
                ubm = torch.le(unified_t, (i+1)*0.1)
                mask_t = torch.where(torch.logical_and(lbm, ubm), 1, 0).to(dtype=torch.int32)
                for j in range(self.key_book.n_e):
                    table[i, j] = ((indices == j).to(torch.int) * mask_t).sum()
            torch.set_printoptions(precision=8, sci_mode=False)
            print(table)

    def loss_reward(self, states, indices):
        # indices (B, T) choice of index of this batch
        # reward (B, T) along with their reward ([0, 1] classifier score)
        B, T = indices.shape
        assert self.sa_type == 'egpt' or self.sa_type == 'egpthn'
        for di in range(self.key_book.n_e):
            self.vq_turn = ((self.vq_turn + 1) % self.key_book.n_e)
            mask_i = torch.eq(indices, self.vq_turn)
            n_i = mask_i.sum()
            if n_i != 0:
                # update the reward function of this part
                key_same_i = self.key_book.vparams(torch.tensor([[self.vq_turn]], device=states.device))
                key_same_i = F.normalize(key_same_i, p=2.0, dim=-1)
                key_same_i = key_same_i.repeat(B, T, 1)
                reward_i, _ = self.sa_net.get_reward(states, key_same_i)
                reward_i = torch.where(mask_i, reward_i, 1.0 - reward_i)  # other should be different
                return reward_i
                # return torch.zeros_like(indices, dtype=torch.float32, device=states.device)

        print('in autocot method: loss_reward: should not reach here!')
        assert False

    def training_step(self, batch, batch_idx):
        # separate optimzing whole net and the embedding space
        self.step += 1
        opt_policy, opt_cluster = self.optimizers()
        sch_policy, sch_cluster = self.lr_schedulers()

        states, timesteps, actions, unified_t, lengths = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
        # zero_ts_mask = torch.eq(unified_t[:, 0], 0)
        unified_t = unified_t[:states.shape[0], :states.shape[1]]

        ##############################################################################
        # first forward, use policy and implicit energy to adjust key_soft, key_book #
        ##############################################################################
        key_soft, state_trans = self.key_net(states, timesteps, actions)
        if self.use_st:
            # use time sphere embedding
            key_soft = self.ks_t_emb(F.normalize(key_soft, p=2.0, dim=-1), unified_t)
        elif self.ks_t_emb is not None:
            key_soft_te = self.ks_t_emb(unified_t)  # (B, T, te_keys_dim)
            key_soft = torch.cat([key_soft, key_soft_te], dim=-1)  # (B, T, key_dim + te_keys_dim)
        state_recs_from_keys = self.rec_net(key_soft, timesteps)

        # loss: state transmission
        l_s_trans, _ = get_loss(state_trans, states[:, 1:, ...], lengths - 1)
        # loss: state reconstruction
        l_s_recs, _ = get_loss(state_recs_from_keys, states, lengths)

        encoding_indices, v_global, _, vparams_hard, w_max, score_vpss_1, score_vpsh_1, score_kss_1, _ \
            = self.key_book(key_soft)

        l_label_cluster = torch.neg(torch.log(w_max)).mean()
        loss_key_soft_adj = 0.0

        if self.sa_type == 'hn':
            # currently hyper net implementation only have
            action_preds = self.sa_net(states, timesteps, keys=vparams_hard)

            # loss: state prediction & action prediction
            l_s_preds = 0.0
            l_a_preds, _ = get_loss(action_preds, actions, lengths)

            # need some regulation on the explicit reward (ascent reward, large at switching point)
            #######
            # ??? #
            #######
            l_policy = l_a_preds + l_s_trans + self.coe_rec * (l_s_recs + loss_key_soft_adj)

        elif self.sa_type == 'egpt' or self.sa_type == 'egpthn':
            action_preds, reward, v_r_norm = self.sa_net(states, timesteps, actions=actions, keys=vparams_hard)

            # loss: state prediction & action prediction
            l_s_preds = 0.0
            l_a_preds, ls_a_preds = get_loss(action_preds, actions, lengths)

            self.ls_a_pred_mean = (self.ls_a_pred_mean * float(self.cnt_a_pred_mean) + l_a_preds) / float(self.cnt_a_pred_mean + 1)
            self.cnt_a_pred_mean += 1

            # update label action prediction losses
            self.key_book.update_loss_label(losses=ls_a_preds, unified_t=unified_t, indices=encoding_indices)

            # need some regulation on the explicit reward (ascent reward, large at switching point)
            l_policy = l_a_preds + l_s_trans + self.coe_rec * (l_s_recs + loss_key_soft_adj)
            if self.flag_cluster:
                # reward is the prob of completion
                # (1.0 - reward) is the prob of un-completion
                # log(1.0 - reward) should be as large as possible
                # minimize -log(1.0 - reward)
                # also we need to update one set of classifier during onr iteration

                # Two direction Lip Constraint
                reward_i = self.loss_reward(states, indices=encoding_indices)

                l_policy = \
                    l_policy \
                    + self.coe_cluster * (
                        l_label_cluster
                        + torch.neg(torch.log(1.0 - reward)).mean()
                        + torch.neg(torch.log(1.0 - reward_i)).mean()
                    )

        else:
            action_preds, state_preds = self.sa_net(states, timesteps, keys=vparams_hard, predict_state=True)

            # loss: state prediction & action prediction
            l_s_preds, _ = get_loss(state_preds[:, :-1, ...], states[:, 1:, ...], lengths - 1)
            l_a_preds, _ = get_loss(action_preds, actions, lengths)
            l_policy = l_s_preds + l_a_preds + l_s_trans + self.coe_rec * (l_s_recs + loss_key_soft_adj)

        opt_policy.zero_grad()
        self.manual_backward(l_policy)
        opt_policy.step()
        sch_policy.step()

        # statistic on encoding indices
        if self.step % 10 == 0:
            self.statistic_indices(encoding_indices, unified_t)

        ###################################################
        # second forward, for classification (clustering) #
        ###################################################
        if (self.sa_type != 'egpt' and self.sa_type != 'egpthn') and self.flag_cluster:
            print('！@#¥%……&*（')
            key_soft, _ = self.key_net(states, timesteps, actions)

            encoding_indices, v_global, _, _, score_vpss_2, score_vpsh_2, score_kss_2, score_ksh_2 \
                = self.key_book(key_soft)

            pn_change_ss = torch.where(torch.less(score_vpss_1, score_vpss_2), 0.0, 1.0)
            pn_change_sh = torch.where(torch.less(score_vpsh_1, score_vpsh_2), 0.0, 1.0)

            logit_vpss = torch.neg(torch.log(F.sigmoid(score_vpss_2)))  # (B, T, T)
            logit_vpsh = torch.neg(torch.log(F.sigmoid(score_vpsh_2)))  # (B, T, n_e)
            logit_kss = torch.neg(torch.log(F.sigmoid(score_kss_2)))  # (B, T, T)
            logit_ksh = torch.neg(torch.log(F.sigmoid(score_ksh_2)))  # (B, T, n_e)

            logit_contrast_ss = torch.abs(logit_vpss - logit_kss) - self.LIP  # (B, T, T)
            logit_contrast_sh = torch.abs(logit_vpsh - logit_ksh) - self.LIP  # (B, T, n_e)

            logit_contrast_ss = logit_contrast_ss * pn_change_ss * self.w_ss
            logit_contrast_sh = logit_contrast_sh * pn_change_sh

            logit_contrast_ss = torch.maximum(logit_contrast_ss, torch.zeros_like(logit_contrast_ss))
            logit_contrast_sh = torch.maximum(logit_contrast_sh, torch.zeros_like(logit_contrast_sh))

            log_logit_contrast_ss = logit_contrast_ss.mean()
            log_logit_contrast_sh = logit_contrast_sh.mean()

            l_cluster = torch.cat([logit_contrast_ss, logit_contrast_sh, logit_contrast_sh], dim=-1)
            l_cluster = torch.mean(l_cluster, dim=2)
            l_cluster = torch.mean(l_cluster, dim=1)
            l_cluster = torch.mean(l_cluster, dim=0)

            opt_cluster.zero_grad()
            self.manual_backward(l_cluster)
            opt_cluster.step()
            sch_cluster.step()  # dont forget schedulers !!!!

        else:
            l_cluster = 0.0
            log_logit_contrast_ss = 0.0
            log_logit_contrast_sh = 0.0

        # log the loss-es
        self.log_dict(
            {
                'vg': v_global,
                'ts': l_s_trans,
                'rs': l_s_recs,
                '?s': l_s_preds,
                '?a': l_a_preds,
                'regks': loss_key_soft_adj,
                '©l': l_label_cluster,
                '©': l_cluster,
                '©ss': log_logit_contrast_ss,
                '©sh': log_logit_contrast_sh,
            }, prog_bar=True, on_step=True, on_epoch=True
        )

    def label_single(self, states, timesteps, unified_t, actions=None):
        # states: (T, s_dim)
        # actions: None or (T - 1, a_dim)
        # timesteps

        # states = states[None, ...]
        # timesteps = timesteps[None, ...]
        # if actions is not None:
        #     actions = actions[None, ...]

        # key_net label
        key_soft, _ = self.key_net(states, timesteps, actions)
        if self.use_st:
            # use time sphere embedding
            key_soft = self.ks_t_emb(F.normalize(key_soft, p=2.0, dim=-1), unified_t)
        elif self.ks_t_emb is not None:
            key_soft_te = self.ks_t_emb(unified_t)  # (B, T, te_keys_dim)
            key_soft = torch.cat([key_soft, key_soft_te], dim=-1)  # (B, T, key_dim + te_keys_dim)
        label = self.key_book.get_key_soft_indices(key_soft)
        return label[:, -1]

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the module into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()  # all parameters which need to decay
        no_decay = set()  # all parameters which do not need to decay

        decay_wo_sa = set()  # all parameters which need to decay, apart from those in sa_net
        no_decay_wo_sa = set()  # all parameters which need to decay, apart from those in sa_net

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                    if not fpn.startswith('sa_net'):
                        no_decay_wo_sa.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                    if not fpn.startswith('sa_net'):
                        decay_wo_sa.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                    if not fpn.startswith('sa_net'):
                        no_decay_wo_sa.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        # NEED MODIFICATION
        no_decay.add('key_net.local_pos_emb')
        no_decay.add('key_net.global_pos_emb')
        no_decay_wo_sa.add('key_net.local_pos_emb')
        no_decay_wo_sa.add('key_net.global_pos_emb')
        no_decay.add('rec_net.local_pos_emb')
        no_decay.add('rec_net.global_pos_emb')
        no_decay_wo_sa.add('rec_net.local_pos_emb')
        no_decay_wo_sa.add('rec_net.global_pos_emb')
        no_decay.add('sa_net.local_pos_emb')
        no_decay.add('sa_net.global_pos_emb')

        # for pn in sorted(list(decay_wo_sa)):
        #     print(pn)
        # for pn in sorted(list(no_decay_wo_sa)):
        #     print(pn)

        # validate that we considered every parameter

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        # # separate code book...
        # no_decay.remove('key_book.embedding.weight')
        # key_book_emb.add('key_book.embedding.weight')
        optim_policy = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": self.optimizers_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optim_cluster = [
            {"params": [param_dict[pn] for pn in sorted(list(decay_wo_sa))],
             "weight_decay": self.optimizers_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay_wo_sa))],
             "weight_decay": 0.0},
        ]
        # optim_groups = [
        #     {"params": [param_dict[pn] for pn in sorted(list(decay))],
        #      "weight_decay": self.optimizers_config['weight_decay']},
        #     {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
        #      "weight_decay": 0.0},
        # ]

        if self.optimizers_config is not None:
            optimizer_policy = torch.optim.AdamW(
                optim_policy,
                lr=self.optimizers_config['init_lr'],
                betas=(self.optimizers_config['beta1'], self.optimizers_config['beta2'])
            )
            optimizer_cluster = torch.optim.AdamW(
                optim_cluster,
                lr=self.optimizers_config['init_lr'],
                betas=(self.optimizers_config['beta1'], self.optimizers_config['beta2'])
            )
        else:
            optimizer_policy = torch.optim.AdamW(optim_policy, lr=1e-4, betas=(0.9, 0.95))
            optimizer_cluster = torch.optim.AdamW(optim_cluster, lr=1e-4, betas=(0.9, 0.95))

        # scheduler config
        if self.scheduler_config is None:
            return optimizer_policy, optimizer_cluster

        assert 'type' in self.scheduler_config.keys()
        if self.scheduler_config['type'] == 'cos_decay_with_warmup':
            assert 't_max', 't_warmup' in self.scheduler_config.keys()
            scheduler_policy = CosineAnnealingLRWarmup(
                optimizer_policy,
                T_max=self.scheduler_config['t_max'],
                T_warmup=self.scheduler_config['t_warmup']
            )
            scheduler_cluster = CosineAnnealingLRWarmup(
                optimizer_cluster,
                T_max=self.scheduler_config['t_max'],
                T_warmup=self.scheduler_config['t_warmup']
            )
        elif self.scheduler_config['type'] == 'multistep':
            assert 'milestones', 'gamma' in self.scheduler_config.keys()
            scheduler_policy = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_policy,
                milestones=self.scheduler_config['milestones'],
                gamma=self.scheduler_config['gamma']
            )
            scheduler_cluster = torch.optim.lr_scheduler.MultiStepLR(
                optimizer_cluster,
                milestones=self.scheduler_config['milestones'],
                gamma=self.scheduler_config['gamma']
            )
        else:
            assert False

        return [optimizer_policy, optimizer_cluster], [scheduler_policy, scheduler_cluster]
