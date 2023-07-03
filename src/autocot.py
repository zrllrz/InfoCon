import numpy as np
from math import exp
import torch
import torch.nn as nn

from module.VQ import VQ2Linear
from module.GPT import KeyNet, ActNet, RecNet

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from lr_scheduler import CosineAnnealingLRWarmup

from einops import rearrange

def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape, f'mse_loss_with_weights: shape mismatch{losses.shape, weights.shape}'
        return torch.mean(losses * weights)


def get_loss(preds, targets, lengths):
    # If we have sequences of varied lengths, use masks so we do not compute loss
    # over padded values. If we set max_seq_length=min_seq_length, then it should
    # not matter since all sequences have the same length.
    B = preds.shape[0]
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    lengths = lengths[:, None]  # B x 1
    temp = torch.arange(0, max_len)[None].expand(B, -1).to(lengths.device)  # B x max_len
    masks = (temp < lengths.expand(B, max_len)).float()  # B x max_len

    loss = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1)
    )
    return loss

def init_centriods(datas, n_centriods):
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    cent_init = datas[i0][None, :]
    for _ in range(n_centriods - 1):
        d = torch.sum(datas ** 2, dim=1, keepdim=True) + \
            torch.sum(cent_init ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', datas, rearrange(cent_init, 'n d -> d n'))
        d_min = d.min(dim=1)[0] + 0.0001
        f_d_min = d_min / d_min.sum()
        i = f_d_min.multinomial(num_samples=1)[0]
        cent_init = torch.cat([cent_init, datas[i][None, :]], dim=0)
    return cent_init


class BaseConfig:
    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop):
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop


class KeyNetConfig(BaseConfig):
    def __init__(self, block_size, n_layer, n_embd, n_head, model_type,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 max_timestep, use_skip_connection):
        super().__init__(block_size, n_embd, n_head, attn_pdrop, resid_pdrop)
        self.n_layer = n_layer
        assert 'cot' not in model_type, 'KeyNet cannot process key states input'
        self.len_key_states = 0
        if '+a' in model_type:
            self.block_size *= 2
        self.model_type = model_type
        self.embd_pdrop = embd_pdrop
        self.max_timestep = max_timestep
        self.use_skip_connection = use_skip_connection


class ActNetConfig(BaseConfig):
    def __init__(self, block_size, n_layer, n_embd, n_head, model_type,
                 attn_pdrop, resid_pdrop, key_states):
        super().__init__(block_size, n_embd, n_head, attn_pdrop, resid_pdrop)
        self.n_layer = n_layer
        assert 'cot' in model_type, 'ActNet MUST have key states prompt input'
        self.model_type = model_type
        if '+a' in model_type:
            self.block_size *= 2

        # It is in the form of 'acd...' that represents whether the key
        # state x is used. e.g., here a,c,d is used while b is skipped.
        assert key_states not in ['', None] and \
               np.all([ord('z') >= ord(g) >= ord('a') for g in key_states])
        self.key_states = key_states
        self.len_key_states = len(key_states)


class RecNetConfig(BaseConfig):
    def __init__(self, block_size, n_layer, n_embd, n_head, model_type,
                 attn_pdrop, resid_pdrop):
        super().__init__(block_size, n_embd, n_head, attn_pdrop, resid_pdrop)
        self.n_layer = n_layer
        assert 'cot' not in model_type and '+a' not in model_type, 'KeyNet cannot process key states input'
        self.len_key_states = 0
        self.model_type = model_type


class AutoCoT(pl.LightningModule):
    def __init__(
        self,
        key_config,
        vq_len,
        vq_beta,
        vq_legacy,
        act_config,
        rec_config,
        vq_log=True,
        vq_kmeans_reset=None,
        vq_kmeans_step=None,
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

        self.key_net = KeyNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        self.key_states_book = VQ2Linear(
            n_e=vq_len,
            e_dim=key_config.n_embd,
            beta=vq_beta,
            legacy=vq_legacy,
            log_choice=vq_log
        )
        self.len_key_states = act_config.len_key_states  # len of prompt for the Act-Net
        self.book_out = nn.Sequential(
            nn.Linear(
                act_config.n_embd,
                act_config.n_embd * act_config.len_key_states
            ),
            nn.SiLU(),
            nn.Linear(
                act_config.n_embd * act_config.len_key_states,
                act_config.n_embd * act_config.len_key_states
            )
        )
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

        self.act_net = ActNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        self.rec_net = RecNet(
            config=rec_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # deal with mode collapse
        self.p_collapse_0 = 0.01
        self.A = 100.0
        self.delta_T = 0

        # training state
        self.bgstate = 0  # 0, 1, 2, 3, 4, 5. 0 for normal state, 5 for collapse state
        self.state_label = 'normal'  # 'normal', 'collapse'

    def encode(self, states, timesteps, actions=None):
        key_emb, x, T, int_feat = self.key_net(states, timesteps, actions)
        return key_emb, x, T, int_feat

    def book_neck(self, key_emb, lengths=None):
        if lengths is not None:
            ind = (lengths[:, None, None] - 1).repeat(1, 1, key_emb.shape[2])
            key_emb_g = key_emb.gather(dim=1, index=ind).squeeze(1)
            key_emb_q, emb_q_loss, indices, v = self.key_states_book(key_emb_g)
        else:
            key_emb_g = key_emb[:, -1]
            key_emb_q, emb_q_loss, indices, v = self.key_states_book(key_emb_g)
        key_emb_out = (self.book_out(key_emb_q)).view(-1, self.len_key_states, self.n_embd)
        return key_emb_out, emb_q_loss, indices, v, key_emb_g

    def decode(self, key_emb_out, x, T, int_feat, key_state_mask=None):
        act_preds = self.act_net(key_emb_out, x, T, int_feat, key_state_mask=key_state_mask)
        return act_preds

    def recons(self, key_emb, T):
        state_recs = self.rec_net(key_emb, T)
        return state_recs

    def collapse_exception(self, batch):
        print('Index collapse :(\nResetting Key Book using K-Means')
        arranged_mask = torch.arange(self.key_states_book.n_e)[:, None]
        arranged_mask = arranged_mask.to(self.device)
        states, timesteps, actions, lengths = batch['s'], batch['t'], batch['a'], batch['lengths']
        key_emb, _, _, _ = self.encode(states, timesteps, actions)
        if '+a' in self.key_net.config.model_type:
            key_emb_rec = key_emb[:, ::2]
        else:
            key_emb_rec = key_emb

        # Re-initial key book
        _, _, indices, _, key_emb_g = self.book_neck(key_emb_rec, lengths=lengths)
        # print('key_emb_g shape:', key_emb_g.shape)
        # print(self.key_states_book.n_e)
        self.key_states_book.embedding.weight.data = init_centriods(key_emb_g, self.key_states_book.n_e)

        # Do kmeans algorithm to reset
        for _ in range(self.vq_kmeans_step):
            _, _, indices, _, key_emb_g = self.book_neck(key_emb_rec, lengths=lengths)
            expanded_indices = indices[None].expand(self.key_states_book.n_e, -1)
            mask = (expanded_indices == arranged_mask).to(key_emb_rec.dtype)
            c_grad = mask @ key_emb_g / mask.sum(-1)[..., :, None]
            torch.nan_to_num_(c_grad)
            # print(c_grad)

            self.key_states_book.embedding.weight.data = c_grad

    # states:    pure vec states (no key states)
    # timesteps: used for the global+local position embedding design
    #            similar to the one in Decision Transformer.
    # actions:   pure vec action (no key action)
    # key_state_mask:
    def forward(self, states, timesteps, actions=None, key_state_mask=None):
        key_emb, x, T, int_feat = self.encode(states, timesteps, actions)
        key_emb_out, emb_q_loss, _, v, _ = self.book_neck(key_emb)
        act_preds = self.decode(key_emb_out, x, T, int_feat, key_state_mask=key_state_mask)
        return act_preds, emb_q_loss

    def training_step(self, batch, batch_idx):
        # Forward pass
        states, timesteps, actions, lengths = batch['s'], batch['t'], batch['a'], batch['lengths']
        key_emb, x, T, int_feat = self.encode(states, timesteps, actions)
        if '+a' in self.key_net.config.model_type:
            key_emb_rec = key_emb[:, ::2]
        else:
            key_emb_rec = key_emb
        key_emb_out, emb_q_loss, _, v, _ = self.book_neck(key_emb_rec, lengths=lengths)
        act_preds = self.decode(key_emb_out, x, T, int_feat, key_state_mask=None)

        state_recs = self.recons(key_emb_rec, T)

        # Obtain training losses
        loss_act_pred = get_loss(act_preds, actions, lengths)
        loss_state_rec = get_loss(state_recs, states, lengths)  # LENGTH?????????
        loss_book_emb = emb_q_loss

        # deal with mode collapse state
        if self.bgstate == 0:
            if v == 0:
                self.bgstate = 1
            self.delta_T += 1
        elif self.bgstate == 5:
            if v != 0:
                self.bgstate = 4
        elif self.bgstate == 1:
            if v == 0:
                self.bgstate = 2
                self.delta_T += (self.state_label == "normal")
            else:
                self.bgstate = 0
                if self.state_label == "normal":
                    self.delta_T += 1
                elif self.state_label == "collapse":
                    self.delta_T = 0
                    self.state_label = 'normal'
        elif self.bgstate == 2:
            if v == 0:
                self.bgstate = 3
                self.delta_T += (self.state_label == "normal")
            else:
                self.bgstate = 1
                self.delta_T += (self.state_label == "normal")
        elif self.bgstate == 3:
            if v == 0:
                self.bgstate = 4
                self.delta_T += (self.state_label == "normal")
            else:
                self.bgstate = 2
                self.delta_T += (self.state_label == "normal")
        elif self.bgstate == 4:
            if v == 0:
                self.bgstate = 5
                self.state_label = 'collapse'
            else:
                self.bgstate = 3
                self.delta_T += (self.state_label == "normal")
        else:
            print('Exception'), 'Not defined bgstate'

        # get out of trap if necessary
        assert self.state_label in ['collapse', 'normal']
        if self.state_label == 'collapse':
            if lengths is not None:
                ind = (lengths[:, None, None] - 1).repeat(1, 1, key_emb.shape[2])
                key_emb_last = key_emb.gather(dim=1, index=ind).squeeze(1)
            else:
                key_emb_last = key_emb_rec[:, -1].detach()
            d = torch.sum(key_emb_last ** 2, dim=1, keepdim=True) + \
                torch.sum(self.key_states_book.embedding.weight ** 2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', key_emb_last, rearrange(self.key_states_book.embedding.weight, 'n d -> d n'))
            loss_collapse = d.mean()
            loss = loss_act_pred + loss_state_rec + loss_collapse
        elif self.state_label == 'normal':
            proba = self.p_collapse_0 * exp(- self.A * self.delta_T)
            if torch.rand(size=[1, ]) < proba:
                if lengths is not None:
                    ind = (lengths[:, None, None] - 1).repeat(1, 1, key_emb.shape[2])
                    key_emb_last = key_emb.gather(dim=1, index=ind).squeeze(1)
                else:
                    key_emb_last = key_emb_rec[:, -1].detach()
                d = torch.sum(key_emb_last ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.key_states_book.embedding.weight ** 2, dim=1) - 2 * \
                    torch.einsum('bd,dn->bn', key_emb_last, rearrange(self.key_states_book.embedding.weight, 'n d -> d n'))
                loss_collapse = 0.001 * d.mean()
            else:
                loss_collapse = 0.0
            loss = loss_act_pred + loss_state_rec + loss_book_emb + loss_collapse

        self.log_dict(
            {
                "loss": loss,
                "loss_act_pred": loss_act_pred,
                "loss_state_rec": loss_state_rec,
                "loss_book_emb": loss_book_emb
            }, prog_bar=True, on_step=True, on_epoch=True
        )
        if v is not None:
            self.log("choice_var", v, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if (self.vq_kmeans_reset is not None) and (self.kmeans_idx % self.vq_kmeans_reset == 0):
            self.collapse_exception(batch)

            self.kmeans_idx += 1

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
