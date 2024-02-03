import sys
sys.path.append("/home/rzliu/AutoCoT/src/module")
from math import cos, pi
import torch
from module.VQ import VQClassifierNNTime
from module.GPT import KeyNet, RecNet, FutureNet, ExplicitSAHNGPT
import pytorch_lightning as pl
from lr_scheduler import CosineAnnealingLRWarmup
from util import get_loss


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


class ExplicitSAHNGPTConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, n_state_layer, use_skip, max_timestep, use_future_state=False):
        self.type = 'egpthn'
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )
        self.n_state_layer = n_state_layer
        self.use_skip = use_skip
        self.use_future_state = use_future_state


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


class FutureNetConfig(RootConfig):
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, n_layer, max_timestep):
        super().__init__(
            n_embd, n_head,
            attn_pdrop, resid_pdrop, embd_pdrop,
            block_size, 'w_key', n_layer, max_timestep
        )


class AutoCoT(pl.LightningModule):
    def __init__(
        self,
        key_config,
        rec_config,
        sa_config,
        future_config=None,
        vq_n_e=10,
        vq_use_r=False,
        vq_coe_ema=0.95,
        vq_ema_ave=False,
        KT=0.1,
        optimizers_config=None,
        scheduler_config=None,
        state_dim=-1,
        action_dim=-1,
        key_dim=-1,
        e_dim=-1,
        vq_use_ft_emb=False,
        vq_use_st_emb=False,
        vq_st_emb_rate=2.0,
        vq_coe_r_l1=0.0,
        vq_use_prob_sel_train=False,
        vq_use_timestep_appeal=False,
        task=''
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

        self.rec_net = RecNet(
            config=rec_config,
            state_dim=state_dim,
            key_dim=key_dim+vq_use_st_emb
        )

        # sa_net, for action, next_state prediction to eval the hard key
        if future_config is None:
            self.future_net = None
        else:
            self.future_net = FutureNet(
                config=future_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=key_dim+vq_use_st_emb
            )

        self.sa_type = sa_config.type
        if sa_config.type == 'egpthn':
            self.sa_net = ExplicitSAHNGPT(
                config=sa_config,
                state_dim=state_dim,
                action_dim=action_dim,
                key_dim=e_dim,
                KT=KT,
            )
        else:
            print('suggest using sa_config.type = \'egpthn\'')
            assert False

        # key_book, use for vq mapping
        # every soft key will be mapped to a hard key in the book using a restrict nearest neighbour
        # which will make sure the indices are close...
        self.key_book = VQClassifierNNTime(
            key_dim=key_dim,
            n_e=vq_n_e,
            e_dim=e_dim * sa_config.n_state_layer if (sa_config.type == 'hn' or sa_config.type == 'egpthn') else e_dim,
            e_split=sa_config.n_state_layer,
            KT=KT,
            use_r=vq_use_r,
            use_ema=True,
            coe_ema=vq_coe_ema,
            ema_ave=vq_ema_ave,
            use_ft_emb=vq_use_ft_emb,
            use_st_emb=vq_use_st_emb,
            t_emb_rate=vq_st_emb_rate,
            use_prob_sel_train=vq_use_prob_sel_train,
            use_timestep_appeal=vq_use_timestep_appeal
        )
        self.vq_coe_r_l1 = vq_coe_r_l1

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

        if optimizers_config is not None:
            self.coe_cluster = optimizers_config['coe_cluster']
            self.coe_rec = optimizers_config['coe_rec']
            self.use_decay_mask_rate = False if 'use_decay_mask_rate' not in optimizers_config.keys() else optimizers_config['use_decay_mask_rate']
        else:
            self.coe_cluster = 0.0
            self.coe_rec = 0.0
            self.use_decay_mask_rate = False

        self.step = 0
        self.step_cluster = 0
        self.cyc_cluster = 1
        self.flag_cluster = False
        self.goal_cluster = True

        self.progress_bar_step = 0.0

        self.automatic_optimization = False
        self.mytask = task

        self.half_linear_increase = 0.0
        self.half_linear_increase_stop = 0.0

        self.mode = 'rec_goal'
        # two mode: 'rec': only train vqvae
        #           'goal': first load vqvae trained checkpoint, then train with goal

    def mask_cluster_rate(self):
        # calculate the mask off rate of cluster according to action prediction behavior
        # return a fp between [0, 1], meaning the proportion of mask off clustering place
        return (0.5 + cos(self.progress_bar_step * pi) * 0.5) ** 10

    def on_train_batch_start(self, batch, batch_idx):
        max_steps = self.trainer.max_steps
        global_step = self.trainer.global_step
        self.progress_bar_step = global_step / max_steps
        self.half_linear_increase = max(0.0, self.progress_bar_step * 2.0 - 1.0)
        self.half_linear_increase_stop = min(1.0, self.progress_bar_step * 2.0)

        self.key_book.reset_oor_r(left_step=(max_steps-global_step+1))

        self.step_cluster += 1
        if self.step_cluster == self.cyc_cluster:
            self.step_cluster = 0
            self.flag_cluster = True
        else:
            self.flag_cluster = False

    def statistic_indices(self, indices, unified_t=None):
        # (B, T)
        print(self.mytask)
        if unified_t is None:
            table = torch.zeros(size=(indices.shape[0], self.key_book.n_e))  # (B, n_e)
            for i in range(self.key_book.n_e):
                table[:, i] = torch.sum((indices == i).to(torch.int), dim=-1)
            print(table[:min(indices.shape[0], 32), ...])
        else:
            table = torch.zeros(size=(self.key_book.n_e, self.key_book.n_e), dtype=torch.int32)  # (B, n_e)
            for i in range(self.key_book.n_e):
                lbm = torch.ge(unified_t, i / self.key_book.n_e)
                ubm = torch.le(unified_t, (i+1) / self.key_book.n_e)
                mask_t = torch.where(torch.logical_and(lbm, ubm), 1, 0).to(dtype=torch.int32)
                for j in range(self.key_book.n_e):
                    table[i, j] = ((indices == j).to(torch.int) * mask_t).sum()
            torch.set_printoptions(precision=8, sci_mode=False)
            print(table)
            print(self.key_book.t_keys.weight.data.squeeze(-1))
            print(self.key_book.get_r().squeeze(-1))
        print('half_linear_increase', self.half_linear_increase)
        print('half_linear_increase_stop', self.half_linear_increase_stop)

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
                key_same_i = self.key_book.split_vparams_norm(key_same_i)
                key_same_i = key_same_i.repeat(B, T, 1)
                reward_i, _ = self.sa_net.get_reward(states, key_same_i)
                reward_i = torch.where(mask_i, reward_i, 1.0 - reward_i)  # other should be different
                return reward_i

        print('in autocot method: loss_reward: should not reach here!')
        assert False

    def get_future_state(self, states, indices):
        # states (B, T, C)
        # indices (B, T)
        B, T = indices.shape
        arange_ext = torch.arange(T, device=states.device).view(1, -1).repeat(B, 1)
        mask = torch.logical_not(torch.eq(indices[:, 1:], indices[:, :-1]))

        last_mask = mask + torch.sum(mask, dim=-1, keepdim=True) - torch.cumsum(mask, dim=-1)
        last_mask = last_mask.to(dtype=torch.bool)
        last_mask = torch.cat([last_mask, torch.zeros(size=(B, 1), dtype=torch.bool, device=states.device)], dim=-1)
        last_mask = last_mask.to(dtype=torch.float32)

        his_ind = torch.full(size=(B, 1), fill_value=(T-1), device=states.device)
        future_state_indices = torch.full(size=(B, 1), fill_value=(T-1), device=states.device)
        for i in range(T-2, -1, -1):
            new_ind = torch.where(mask[:, i:(i+1)], arange_ext[:, i:(i+1)], his_ind)
            his_ind = new_ind
            future_state_indices = torch.cat([new_ind, future_state_indices], dim=1)
        future_state = \
            torch.gather(states, dim=1, index=future_state_indices.unsqueeze(-1).repeat(1, 1, states.shape[2]))

        return future_state, last_mask

    def rec_training_step(self, batch, batch_idx):
        self.step += 1
        opt_policy, opt_cluster = self.optimizers()
        sch_policy, sch_cluster = self.lr_schedulers()

        states, timesteps, actions, unified_t, lengths = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
        unified_t = unified_t[:states.shape[0], :states.shape[1]]

        key_soft, state_trans = self.key_net(states, timesteps, actions)
        encoding_indices, key_hard, _, _, w_max, w_cnt, _ = \
            self.key_book(key_soft, unified_t)

        # only do the encoding, which is a start point for concept extraction
        state_recs_from_keys = self.rec_net(key_hard, timesteps)
        l_s_recs, _ = get_loss(state_recs_from_keys, states, lengths)

        # clustering behind
        enough_mask = torch.less(w_max, 0.5 + 0.5 * self.half_linear_increase_stop).to(dtype=torch.float32)
        ls_label_cluster = torch.neg(torch.log(w_max))
        l_label_cluster = ls_label_cluster.mean()  # for log

        ls_label_cluster_masked = ls_label_cluster * enough_mask
        ls_label_cluster_weighed = ls_label_cluster_masked * w_cnt

        l_label_cluster_weighed = torch.div(ls_label_cluster_weighed.sum(), self.key_book.n_e)

        l_rec = l_s_recs + self.coe_cluster * l_label_cluster_weighed

        opt_policy.zero_grad()
        self.manual_backward(l_rec)
        opt_policy.step()
        sch_policy.step()

        # statistic on encoding indices
        if self.step % 10 == 0:
            self.statistic_indices(encoding_indices, unified_t)

        self.log_dict(
            {
                'rs': l_s_recs,
                '©l': l_label_cluster,
            }, prog_bar=True, on_step=True, on_epoch=True
        )

    def training_step(self, batch, batch_idx):
        # separate optimzing whole net and the embedding space

        if self.mode == 'rec':
            return self.rec_training_step(batch, batch_idx)

        assert self.mode == 'goal'

        self.step += 1
        opt_policy, opt_cluster = self.optimizers()
        sch_policy, sch_cluster = self.lr_schedulers()

        states, timesteps, actions, unified_t, lengths = batch['s'], batch['t'], batch['a'], batch['unified_t'], batch['lengths']
        # zero_ts_mask = torch.eq(unified_t[:, 0], 0)
        unified_t = unified_t[:states.shape[0], :states.shape[1]]

        key_soft, state_trans = self.key_net(states, timesteps, actions)
        encoding_indices, key_hard, vparams_w, vparams_hard, w_max, w_cnt, key_soft_t_emb = \
            self.key_book(key_soft, unified_t)

        coe_future = self.half_linear_increase_stop

        if self.future_net is not None and coe_future > 0.0:
            state_future_preds = self.future_net(states, timesteps, actions, keys=key_hard)
            state_future, last_mask = self.get_future_state(states, encoding_indices)
            l_future, ls_future = get_loss(state_future_preds, state_future, lengths)
            l_future = torch.mean(ls_future * last_mask)
        else:
            l_future = 0.0
            state_future = None
        state_recs_from_keys = self.rec_net(key_hard, timesteps)


        # loss: state transmission
        l_s_trans, _ = get_loss(state_trans, states[:, 1:, ...], lengths - 1)
        # loss: state reconstruction
        l_s_recs, _ = get_loss(state_recs_from_keys, states, lengths)

        # loss: clustering behind
        mask_cluster_rate = self.mask_cluster_rate()
        enough_mask = torch.less(w_max, 0.5 + 0.5 * self.half_linear_increase_stop).to(dtype=torch.float32)
        ls_label_cluster = torch.neg(torch.log(w_max))
        l_label_cluster = ls_label_cluster.mean()

        # loss_key_soft_adj = 0.0
        assert self.sa_type == 'egpthn'
        action_preds, reward, v_r_norm = self.sa_net(states, timesteps, actions=actions, keys=vparams_hard,
                                                     future_states=state_future)

        # loss: state prediction & action prediction
        l_s_preds = 0.0
        l_a_preds, ls_a_preds = get_loss(action_preds, actions, lengths)

        # select a part of action prediction, only update a part of clustering related loss according to it
        if self.use_decay_mask_rate:
            ls_a_preds_flattened = ls_a_preds.view(-1)
            _, cluster_mask_index = \
                torch.topk(ls_a_preds_flattened, k=int(mask_cluster_rate * ls_a_preds_flattened.shape[0]))
            cluster_mask_flattened = torch.ones_like(ls_a_preds_flattened, dtype=torch.float32)
            cluster_mask_flattened[cluster_mask_index] = 0.0
            cluster_mask = cluster_mask_flattened.reshape(ls_a_preds.shape)
        else:
            mask_cluster_rate = 0.0
            cluster_mask = None

        self.log(name='cm', value=mask_cluster_rate, prog_bar=True, on_step=True, on_epoch=True)

        # need some regulation on the explicit reward (ascent reward, large at switching point)
        l_policy = l_a_preds + coe_future * l_future + self.coe_rec * l_s_recs
        if self.flag_cluster:
            # reward is the prob of completion
            # (1.0 - reward) is the prob of un-completion
            # log(1.0 - reward) should be as large as possible
            # minimize -log(1.0 - reward)
            # also we need to update one set of classifier during onr iteration
            # print('before reward_i')
            # print('encoding_indices', encoding_indices)
            reward_i = self.loss_reward(states, indices=encoding_indices)
            self.log(name='rt', value=float(self.vq_turn), prog_bar=True, on_step=True, on_epoch=True)

            ls_label_cluster_masked = ls_label_cluster * enough_mask
            if self.use_decay_mask_rate:
                ls_label_cluster_masked = ls_label_cluster_masked * cluster_mask
            ls_label_cluster_weighed = ls_label_cluster_masked * w_cnt

            ls_reward = torch.neg(torch.log(1.0 - reward))
            ls_reward_i = torch.neg(torch.log(1.0 - reward_i))
            if self.use_decay_mask_rate:
                ls_reward = ls_reward * cluster_mask
                ls_reward_i = ls_reward_i * cluster_mask
            ls_reward_weighed = ls_reward * w_cnt

            l_policy = \
                l_policy \
                + self.coe_cluster * (
                    torch.div(ls_label_cluster_weighed.sum(), self.key_book.n_e)
                    + torch.div(ls_reward_weighed.sum(), self.key_book.n_e)
                    + ls_reward_i.mean()
                )

        opt_policy.zero_grad()
        self.manual_backward(l_policy)
        opt_policy.step()
        sch_policy.step()

        # statistic on encoding indices
        if self.step % 10 == 0:
            self.statistic_indices(encoding_indices, unified_t)

        # log the loss-es

        self.log_dict(
            {
                'fu': l_future,
                'ts': l_s_trans,
                'rs': l_s_recs,
                '?s': l_s_preds,
                '?a': l_a_preds,
                '©l': l_label_cluster,
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
        label = self.key_book.get_key_soft_indices(key_soft, unified_t)
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
                    print(fpn)
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
        no_decay.add('sa_net.local_pos_emb')
        no_decay.add('sa_net.global_pos_emb')
        if self.future_net is not None:
            no_decay.add('future_net.local_pos_emb')
            no_decay.add('future_net.global_pos_emb')
            no_decay_wo_sa.add('future_net.local_pos_emb')
            no_decay_wo_sa.add('future_net.global_pos_emb')

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
