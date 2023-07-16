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


class BasicCausalSelfAttention(nn.Module):
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
        block_size = config.block_size * 2
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


class ActCausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention layers for ActNet
    Need to deal with s(0: t-1), a(0: t-2), k(0: t-1), 3 * config.block_size real block_size
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # dropouts
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        block_size = config.block_size * 3

        # action can choose to use only k(t-1) or k[0:t-1]
        # It is a tricky mask, see docs for more details
        mask1 = torch.repeat_interleave(torch.tril(torch.ones(config.block_size, config.block_size)), 3, dim=0)
        if config.seq_k is True:
            mask2 = torch.tril(torch.ones(config.block_size, config.block_size))
        else:
            mask2 = torch.eye(config.block_size)
        mask = torch.zeros(size=(block_size, block_size))
        mask[:, ::3] = mask1
        mask[1::3, 1::3] = mask2
        mask[2:, 2::3] = mask1[:-2, :]
        # print(mask[:9, :9])
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

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
        select it to be:
            BasicCausalSelfAttention or ActCausalSelfAttention
        with config.block_type {'basic', 'act'}
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.block_type == 'basic':
            self.attn = BasicCausalSelfAttention(config)
        elif config.block_type == 'act':
            self.attn = ActCausalSelfAttention(config)
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


class BasicNet(nn.Module):
    """
    GPT Encoder.
    In our method, we use it as KeyNet to:
        encode s[0:(t-1)], a[0:(t-1)] into k_emb[0:(t-1)]
        If set config.do_commit True, it also:
            encode s[0:(t-1)], a[0:(t-2)] into k_commit[0:(t-1)]
    We can also use it to:
        encode s[0:(t-1)], a[0:(t-2)] into k_commit[0:(t-1)]
    Independently.
    """
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.block_type == 'basic'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.block_size = config.block_size * 2  # state + action

        self.do_commit = config.do_commit  # if True, label will be predicted here
        self.sub_pos = config.sub_pos  # if True, we will subtract pos embedding for key prediction

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
        # Key predictor
        self.key_predictor = MLP(config.n_embd, config.n_embd, hidden_dims=[256, 256])

        # print('init module in BasicNet')
        self.apply(self._init_weights)
        # print('init module in BasicNet done')

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

        key_emb = self.drop(x)
        key_emb, _ = self.blocks(key_emb)
        key_emb = self.ln(key_emb)
        key_emb = self.key_predictor(key_emb)

        # do you need to minus the global & loacl embedding here???

        key_soft = key_emb[:, 1:(2*T):2, :]
        if self.sub_pos:
            key_soft = key_soft - pos_emb[:, 1:(2*T):2, :]

        # Return: key_soft, key_commit
        if self.do_commit is False:
            return key_soft, None
        else:
            key_commit_soft = key_emb[:, 0:(2*T):2, :]
            if self.sub_pos:
                key_commit_soft = key_commit_soft - pos_emb[:, 0:(2*T):2, :]
            return key_soft, key_commit_soft


class ExampleNet(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.block_type == 'basic'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.key_dim = config.n_embd
        self.block_size = config.block_size * 2  # keys * 2 in time dim, state + action

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size * 2, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # Key predictor
        self.key_encoder = MLP(config.n_embd, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # State embeddings & Action embeddings
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[256, 256])
        self.action_predictor = MLP(config.n_embd, action_dim, hidden_dims=[256, 256])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Given keys, reconstruction states and actions
    # The process is like when you have a set of method
    # you can put forward a situation using it
    def forward(self, key_hard, timesteps):
        B, T = key_hard.shape[0], key_hard.shape[1]
        key_embeddings = self.key_encoder(key_hard)

        # Token embeddings (B, T*2, n_embd)
        token_embeddings = torch.repeat_interleave(key_embeddings, 2, dim=1)

        # Set up position embeddings.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D

        pos_emb = global_pos_emb + self.local_pos_emb
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        sa = self.ln(x)
        state_preds = self.state_predictor(sa[:, 0:T*2:2, :])
        if T > 1:
            act_preds = self.action_predictor(sa[:, 1:T*2-1:2, :])
        else:
            act_preds = None

        return state_preds, act_preds


class ActNet(nn.Module):
    """
    In our method, we use ActNet to:
        decode k[0:(t-1)], s[0:(t-1)], a[0:(t-2)] into a[0:(t-1)]
        If set config.do_commit True, it also:
            encode s[0:(t-1)], a[0:(t-2)] into k_commit[0:(t-1)]
    """
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.block_type == 'act'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.block_size = config.block_size * 3

        self.do_commit = config.do_commit
        self.sub_pos = config.sub_pos  # if True, we will subtract pos embedding for key prediction

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

        # Key predictor
        if config.do_commit:
            self.key_predictor = MLP(config.n_embd, config.n_embd, hidden_dims=[256, 256])

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

    def forward(self, states, timesteps, actions=None, key_hard=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)

        token_embeddings = torch.zeros([B, T*3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T*3):3, :] = state_embeddings
        if key_hard is not None:   # training
            key_embeddings = self.key_encoder(key_hard)
            token_embeddings[:, 1:(T*3):3, :] = key_embeddings
        # else your are trying to label k(t-1)

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

        if key_hard is None:
            # labeling, make sure you use commitment, return keys
            assert self.do_commit is True
            key_commit_soft = self.key_predictor(x[:, 0:(T*3):3, :])
            if self.sub_pos:
                key_commit_soft = key_commit_soft - pos_emb[:, 0:(T*3):3, :]
            return None, key_commit_soft
        else:
            # training, always need to return prediction of action
            act_preds = self.action_predictor(x[:, 1:(T*3):3, :])
            if self.do_commit is False:
                return act_preds, None
            else:
                key_commit_soft = self.key_predictor(x[:, 0:(T*3):3, :])
                if self.sub_pos:
                    key_commit_soft = key_commit_soft - pos_emb[:, 0:(T*3):3, :]
                return act_preds, key_commit_soft
