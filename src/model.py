import torch
import torch.nn as nn

from .module.VQ import VQKeyState, VQ2Linear
from .module.GPT import KeyNet, ActNet


class AutoCoT(nn.Module):
    def __init__(
        self,
        key_config,
        book_config,
        act_config,
        state_dim=-1,
        action_dim=-1
    ):
        super().__init__()

        assert state_dim > 0 and action_dim > 0

        self.key_net = KeyNet(
            config=key_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

        self.key_states_book = VQ2Linear(
            n_e=book_config.len_book,
            e_dim=book_config.n_embed,
            beta=book_config.vq_beta,
            legacy=book_config.vq_legacy
        )

        self.len_key_states = act_config.len_key_states  # len of prompt for the Act-Net

        self.book_out = nn.Sequential(
            nn.Linear(
                act_config.n_embed,
                act_config.n_embed * act_config.len_key_states
            ),
            nn.SiLU(),
            nn.Linear(
                act_config.n_embed * act_config.len_key_states,
                act_config.n_embed * act_config.len_key_states
            )
        )

        self.act_net = ActNet(
            config=act_config,
            state_dim=state_dim,
            action_dim=action_dim
        )

    # states:    pure vec states (no key states)
    # timesteps: used for the global+local position embedding design
    #            similar to the one in Decision Transformer.
    # actions:   pure vec action (no key action)
    # key_state_mask:
    def forward(self, states, timesteps, actions=None, key_state_mask=None):
        bs = states.shape[0]

        key_emb, x, T = self.key_net(states, timesteps, actions)

        key_emb_q, emb_q_loss = self.key_states_book(key_emb)
        key_emb_out = (self.book_out(key_emb_q)).view(bs, self.len_key_states, -1)

        act_preds = self.act_net(key_emb_out, x, T, key_state_mask=key_state_mask)

        return act_preds, emb_q_loss
