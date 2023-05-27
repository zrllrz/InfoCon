import numpy as np
import torch
import torch.nn as nn

from .module.VQ import VQKeyState, VQ2Linear
from .module.GPT import KeyNet, ActNet

import pytorch_lightning as pl

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


class BaseConfig:
    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop):
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop


class KeyNetConfig(BaseConfig):
    def __init__(self, block_size, n_embd, n_head, model_type, attn_pdrop, resid_pdrop, embd_pdrop, max_timestep):
        super().__init__(block_size, n_embd, n_head, attn_pdrop, resid_pdrop)
        assert 'cot' not in model_type, 'KeyNet cannot process key states input'
        if '+a' in model_type:
            self.block_size *= 2
        self.model_type = model_type
        self.embd_pdrop = embd_pdrop
        self.max_timestep = max_timestep


class ActNetConfig(BaseConfig):
    def __init__(
        self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size,
        model_type, key_states
    ):
        super().__init__(n_embd, n_head, attn_pdrop, resid_pdrop, block_size)
        assert 'cot' in model_type, 'ActNet MUST have key states prompt input'
        self.model_type = model_type
        if '+a' in model_type:
            self.block_size *= 2

        # It is in the form of 'acd...' that represents whether the key
        # state x is used. e.g., here a,c,d is used while b is skipped.
        assert key_states not in ['', None] and \
               np.all([ord('z') >= ord(g) >= ord('a') for g in key_states])

        self.len_key_states = len(key_states)


class AutoCoT(pl.LightningModule):
    def __init__(
        self,
        key_config,
        len_book,
        vq_beta,
        vq_legacy,
        act_config,
        optimizers_config,
        state_dim=-1,
        action_dim=-1
    ):
        super().__init__()

        self.optimizers_config = optimizers_config

        assert state_dim > 0 and action_dim > 0

        self.n_embd = key_config.n_embd

        self.key_net = KeyNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        self.key_states_book = VQ2Linear(
            n_e=len_book,
            e_dim=key_config.n_embd,
            beta=vq_beta,
            legacy=vq_legacy
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

        self.act_net = ActNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

    def encode(self, states, timesteps, actions=None):
        key_emb, x, T = self.key_net(states, timesteps, actions)
        return key_emb, x, T

    def book_neck(self, key_emb):
        key_emb_q, emb_q_loss = self.key_states_book(key_emb)
        key_emb_out = (self.book_out(key_emb_q)).view(-1, self.len_key_states, self.n_embd)
        return key_emb_out, emb_q_loss

    def decode(self, key_emb_out, x, T, key_state_mask=None):
        act_preds = self.act_net(key_emb_out, x, T, key_state_mask=key_state_mask)
        return act_preds

    # states:    pure vec states (no key states)
    # timesteps: used for the global+local position embedding design
    #            similar to the one in Decision Transformer.
    # actions:   pure vec action (no key action)
    # key_state_mask:
    def forward(self, states, timesteps, actions=None, key_state_mask=None):
        key_emb, x, T = self.encode(states, timesteps, actions)
        key_emb_out, emb_q_loss = self.book_neck(key_emb)
        act_preds = self.decode(key_emb_out, x, T, key_state_mask=key_state_mask)
        return act_preds, emb_q_loss

    def training_step(self, batch, batch_idx):
        # Forward pass
        states, timesteps, actions, lengths = batch['s'], batch['t'], batch['a'], batch['lenghts']
        key_emb, x, T = self.encode(states, timesteps, actions)
        key_emb_out, emb_q_loss = self.book_neck(key_emb)
        act_preds = self.decode(key_emb_out, x, T, key_state_mask=None)

        # Obtain training losses
        loss_act_pred = get_loss(act_preds, actions, lengths)
        loss_book_emb = emb_q_loss

        loss = loss_act_pred + loss_book_emb
        return loss

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
                print(fpn)

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    print('\tbias no decay')
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    print('\tweight in white list, decay')
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    print('\tweight in black list, no decay')
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # NEED MODIFICATION
        no_decay.add('local_pos_emb')
        no_decay.add('global_pos_emb')
        if 'cot' in self.model_type:
            no_decay.add('key_state_pos_emb')

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
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.optimizers_config['init_lr'],
            betas=(self.optimizers_config['beta1'], self.optimizers_config['beta2'])
        )
        return optimizer
