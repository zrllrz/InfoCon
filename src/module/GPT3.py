"""
Code for the module architecture of CoTPC, based on the GPT implementation.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
"""

import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], act_fn='relu'):
        super().__init__()
        assert act_fn in ['relu', 'tanh', None, '']
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, j in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(i, j))
            if act_fn == 'relu':
                layers.append(nn.ReLU())
            if act_fn == 'tanh':
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class CausalSelfAttention(nn.Module):
    """
    A basic multi-head masked self-attention layer
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if config.block_type == 'w_key':
            block_size = config.block_size * 3
        elif config.block_type == 'wo_key':
            block_size = config.block_size * 2
        else:
            block_size = config.block_size
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Masked attention

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    A Transformer block,
        select it to be: w_key, wo_key (other will become a more simple self-attn... without action...)
        with config.block_type {'w_key', 'wo_key'}
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BlockLayers(nn.Module):
    """
    A wrapper class for a sequence of Transformer blocks
        select it to be:
            BasicCausalSelfAttention or ActCausalSelfAttention
        with config.block_type {'basic', 'act'}
    """

    def __init__(self, config):
        super().__init__()
        # Register all the individual blocks.
        self.block_list = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.n_head = config.n_head

    def forward(self, x):
        # B, T, _ = x.shape
        output = []  # Also keep the intermediate results.

        for block in self.block_list:
            x = block(x)
            output.append(x)

        return x, output


class KeyCommitNet(nn.Module):
    """
    If using it as ActNet, we
        Recognition of key states from (raw) observation of s[0:(t-1)], a[0:(t-1)]
        encode s[0:(t-1)], a[0:(t-1)] into p_change[0:(t-1)]
        to indicate key state changes
    If using it as CommitNet, we
        decode s[0:(t-1)], a[0:(t-2)] into p_change[0:(t-1)]
        to commit the recognition from KeyNet
    """
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.block_type == 'wo_key'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.block_size = config.block_size * 2  # state + action

        self.do_commit = config.do_commit

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings
        self.state_encoder = MLP(self.state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(self.action_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)

        self.ln = nn.LayerNorm(config.n_embd)
        # - log ( 1 - p ) for switching key state
        # using relu will make sure that output is >= 0, (while - log ( 1 - p ) >= 0)
        self.p_change_predictor = MLP(config.n_embd, 1, hidden_dims=[256, 256])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Given state and action history, put forward possible k_soft[0:t-1]
    # `timesteps` is used for the global+local position embedding
    # Maybe will output commit if needed
    def forward(self, states, timesteps, actions=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)

        # Embeddings for state (action, and key state query) tokens.
        token_embeddings = torch.zeros([B, T*2, self.config.n_embd],
                                       dtype=torch.float32, device=states.device)

        # If using action history as inputs: during training, all actions are
        # specified; during inference, only actions in the past are specified.
        # That is, the first action prediction has no action history as inputs.
        token_embeddings[:, :(T*2):2, :] = state_embeddings

        if actions is not None:
            # Assume the last action is not used as inputs during training.
            if actions.shape[1] >= T:
                action_embeddings = self.action_encoder(actions[:, :T])
                token_embeddings[:, 1:(T*2):2, :] = action_embeddings
            else:
                # last states situation
                assert actions.shape[1] == (T - 1)
                action_embeddings = self.action_encoder(actions[:, :(T-1)])
                token_embeddings[:, 1:(T*2-1):2, :] = action_embeddings

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 2, dim=1)
        pos_emb = global_pos_emb + local_pos_emb

        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        x = self.ln(x)
        if self.do_commit:
            x_key = x[:, 0:(T*2):2, :]  # from s[0:(t-1)], a[0:(t-2)]
        else:
            x_key = x[:, 1:(T*2):2, :]  # from s[0:(t-1)], a[0:(t-1)]

        # - log ( 1 - p )
        minus_log_one_minus_p = self.p_change_predictor(x_key)

        # convert to probability
        p_change = torch.clip(1.0 - torch.exp(-minus_log_one_minus_p), min=0.0, max=1.0)
        print(p_change.shape, minus_log_one_minus_p.shape)

        # return original output ( - log ( 1 - p ) ), and, probability
        return minus_log_one_minus_p, p_change


class ActNet(nn.Module):
    """
    Using it as ActNet, we
        decode s[0:(t-1)], k[0:(t-1)], a[0:(t-2)] into a[0:(t-1)], for prediction
    """
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.block_type == 'w_key'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.block_size = config.block_size * 3

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])
        self.key_encoder = MLP(config.n_embd, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # Action predictor
        self.action_predictor = MLP(config.n_embd, action_dim, hidden_dims=[256, 256])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, timesteps, actions=None, key=None):
        B, T = states.shape[0], states.shape[1]

        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T*3, self.config.n_embd], dtype=torch.float32, device=states.device)
        token_embeddings[:, 0:(T*3):3, :] = state_embeddings
        if key is not None:   # training
            key_embeddings = self.key_encoder(key)
            token_embeddings[:, 1:(T*3):3, :] = key_embeddings
        if actions is not None:  # actions is None when at s0
            # the last action is not used as inputs during ActNet training.
            token_embeddings[:, 2:(T*3-1):3, :] = self.action_encoder(actions[:, :(T-1)])

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 3, dim=1)

        pos_emb = global_pos_emb + local_pos_emb
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        x = self.ln(x)

        x_action = x[:, 1:(T*3):3, :]
        act_preds = self.action_predictor(x_action)
        return act_preds
