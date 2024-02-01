"""
Code for the module architecture of CoTPC, based on the GPT implementation.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
"""

import sys
sys.path.append("/home/rzliu/AutoCoT/src/module")
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from module_util import MLP, CondResnetBlockFC


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class CausalSelfAttention(nn.Module):
    """
    Self Attn Layer, can choose to be of 3 * blocksize or 2 * blocksize
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

        if config.attn_type == 'w_key':
            block_size = config.block_size * 3
        elif config.attn_type == 'wo_key':
            block_size = config.block_size * 2
        else:
            block_size = config.block_size
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
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

    def forward(self, x, skip_feature=None):
        # B, T, _ = x.shape
        output = []  # Also keep the intermediate results.

        if skip_feature is None:
            for block in self.block_list:
                x = block(x)
                output.append(x)
            return x, output

        else:
            for block in self.block_list:
                x = block(x) + skip_feature.pop()
                output.append(x)
            return x, output


class KeyNet(nn.Module):
    """
    KeyNet
    We try to recognize k_s[0:(t-1)] from s[0:(t-1)], a[0:(t-2)].
    """
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'wo_key'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.block_size = config.block_size * 2  # state + action

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)

        self.ln = nn.LayerNorm(config.n_embd)

        # Key soft predictor & state predictor
        self.key_predictor = MLP(config.n_embd, key_dim, hidden_dims=[256])
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[256, 256])

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

    def forward(self, states, timesteps, actions=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)

        # Embeddings for state (action, and key state query) tokens.
        token_embeddings = torch.zeros([B, T*2, self.config.n_embd],
                                       dtype=torch.float32,
                                       device=states.device)

        # states embeddings in token
        token_embeddings[:, :(T*2):2, :] = state_embeddings

        if actions is not None:
            # action embeddings in token
            assert actions.shape[1] >= (T - 1)
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
        x, skip_features = self.blocks(x)
        x = self.ln(x)

        key_soft = self.key_predictor(x[:, 0:(2*T):2, :])
        # only reconstruct s1, s2, ..., s(T-1)
        state_trans = self.state_predictor(x[:, 1:(2*T-2):2, :])

        return key_soft, state_trans


class ImplicitSAGPT(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim

        self.block_size = config.block_size * 3

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])
        self.key_encoder = MLP(key_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln_state = nn.LayerNorm(config.n_embd)
        self.ln_action = nn.LayerNorm(config.n_embd)

        assert config.state_layer < config.n_layer
        self.state_layer = config.state_layer  # which layer to predict state

        # Action predictor
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[256, 256])
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

    def forward(self, states, timesteps, actions=None, keys=None, predict_state=False):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T * 3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T * 3):3, :] = state_embeddings
        if keys is not None:
            # keys should not be None (???????)
            key_embeddings = self.key_encoder(keys)
            token_embeddings[:, 1:(T * 3):3, :] = key_embeddings

        if actions is not None:
            # actions is None when at s0
            # the last action is not used as inputs during ActNet training.
            token_embeddings[:, 2:(T * 3 - 1):3, :] = self.action_encoder(actions[:, :(T - 1)])

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 3, dim=1)

        pos_emb = global_pos_emb + local_pos_emb
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, mid_feature = self.blocks(x)

        if predict_state:
            # predict next state (implicitly modeling the score function of energy behind...)
            x_state = self.ln_state(mid_feature[self.state_layer])
            state_preds = self.state_predictor(x_state[:, 1:(T * 3):3, :])
        else:
            state_preds = None

        # Always predict next action
        x_action = self.ln_action(x)
        action_preds = self.action_predictor(x_action[:, 1:(T * 3):3, :])
        # Use it as ActNet, do action prediction
        return action_preds, state_preds


class ExplicitSAGPT(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim

        self.block_size = config.block_size * 3

        self.n_state_layer = config.n_state_layer  # numbers of layers for reward generation

        # layers to get state grad
        self.state_encoder_0 = MLP(state_dim, config.n_embd, hidden_dims=[])

        self.reward_block = \
            nn.ModuleList(
                CondResnetBlockFC(config.n_embd, key_dim, beta=0) for _ in range(config.n_state_layer)
            )
        self.state_decoder = nn.Linear(config.n_embd, key_dim, bias=False)

        # Below tries to use the gradient info to get policy (action prediction)
        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.state_grad_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])

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

    def get_reward(self, states, keys=None):
        assert keys is not None
        x_states = self.state_encoder_0(states)  # The judgement is unrelated to time...

        for i in range(self.n_state_layer):
            x_states = self.reward_block[i](x_states, keys, last_activation=(i == (self.n_state_layer-1)))
        x_r = self.state_decoder(x_states)  # (B, T, key_dim)
        x_r_norm = F.normalize(x_r, p=2.0, dim=-1)  # (B, T, key_dim)
        r = F.sigmoid(F.cosine_similarity(keys, x_r, dim=-1))
        return r, x_r_norm

    def forward(self, states, timesteps, actions=None, keys=None):
        B, T = states.shape[0], states.shape[1]
        states.requires_grad = True

        # Get state gradient
        r, v_r_norm = self.get_reward(states, keys)
        r_sum = r.sum()
        states_grad = torch.autograd.grad(r_sum, states, retain_graph=True, create_graph=True)[0]

        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T * 3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T * 3):3, :] = state_embeddings
        if keys is not None:
            # keys should not be None (???????)
            grad_embeddings = self.state_grad_encoder(states_grad)
            token_embeddings[:, 1:(T * 3):3, :] = grad_embeddings

        if actions is not None:
            # actions is None when at s0
            # the last action is not used as inputs during ActNet training.
            token_embeddings[:, 2:(T * 3 - 1):3, :] = self.action_encoder(actions[:, :(T - 1)])

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 3, dim=1)

        pos_emb = global_pos_emb + local_pos_emb
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, mid_feature = self.blocks(x)

        # Always predict next action
        x_action = self.ln(x)
        action_preds = self.action_predictor(x_action[:, 1:(T * 3):3, :])
        # Use it as ActNet, do action prediction
        return action_preds, r, v_r_norm


class ExplicitSAHNGPT(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1, KT=0.1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.KT = KT
        self.n_embd = config.n_embd

        self.block_size = config.block_size * 3

        self.n_state_layer = config.n_state_layer  # numbers of layers for reward generation

        # layers to get state grad
        self.state_encoder_0 = nn.Linear(state_dim, config.n_embd)
        # hypernet part
        self.hn_0 = nn.ModuleList(
            nn.Linear(key_dim, config.n_state_layer, bias=False) for _ in range(config.n_state_layer - 1)
        )
        # NOTICE: We SUPPOSE that your input KEY DIM is actually (key_dim * config.n_state_layer)
        self.hn_hidden_1 = nn.Linear(config.n_state_layer, config.n_embd * (config.n_embd + 1), bias=False)
        self.hn_out_1 = nn.Linear(key_dim, config.n_embd, bias=False)
        self.use_skip = config.use_skip  # whether to use skip connection in output network
        self.activation = nn.ReLU()

        # Below tries to use the gradient info to get policy (action prediction)
        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.use_future_state = config.use_future_state
        if config.use_future_state:
            self.state_grad_encoder = MLP(state_dim * 2, config.n_embd, hidden_dims=[256])
        else:
            self.state_grad_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])

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

    # Try to use HyperNet to generate a network
    def get_reward(self, states, keys=None):
        # states (B, T, state_dim)
        # keys (B, T, key_dim * n_state_layer)
        assert keys is not None
        B, T, _ = states.shape
        x_states = self.state_encoder_0(states)  # The judgement is unrelated to time...

        x_hidden = x_states  # (B, T, n_embd)
        x_remain = x_states  # (B, T, n_embd)
        cnt_skip = 0

        for i in range(self.n_state_layer - 1):
            cnt_skip += int(self.use_skip)
            key_i = keys[..., i*self.key_dim:(i+1)*self.key_dim]  # (B, T, key_dim)
            # print('key_i.shape', key_i.shape)
            whn_i = self.hn_0[i](key_i)  # (B, T, self.n_state_layer)
            # print('whn_i.shape', whn_i.shape)
            params_i = self.hn_hidden_1(whn_i)  # (B, T, n_embd * (n_embd + 1))
            # print('params_i.shape', params_i.shape)
            w_i = params_i[..., :(-self.n_embd)].view(B, T, self.n_embd, self.n_embd)  # (B, T, n_embd, n_embd)
            # print('w_i.shape', w_i.shape)
            b_i = params_i[..., (-self.n_embd):]  # (B, T, n_embd)
            # print('b_i.shape', b_i.shape)
            x_hidden = (x_hidden.unsqueeze(2) @ w_i).squeeze(2)  # (B, T, n_embd)
            # print('x_hidden.shape', x_hidden.shape)
            x_hidden = b_i + x_hidden
            # print('x_hidden.shape', x_hidden.shape)
            if cnt_skip == 2:  # always 2 with skip
                x_hidden = x_hidden + x_remain  # skip connection
                # print('x_hidden.shape', x_hidden.shape)
                if i != (self.n_state_layer - 2):
                    x_hidden = self.activation(x_hidden)
                    # print('x_hidden.shape', x_hidden.shape)
                x_remain = x_hidden  # update remain
                # print('x_remain.shape', x_hidden.shape)
                cnt_skip = 0
            else:
                if i != (self.n_state_layer - 2):
                    x_hidden = self.activation(x_hidden)
                    # print('x_hidden.shape', x_hidden.shape)

        v_r_norm = F.normalize(x_hidden, p=2.0, dim=-1)
        # last layer
        key_last = keys[..., (-self.key_dim):]  # (B, T, key_dim)
        mu_last = self.hn_out_1(key_last)  # (B, T, n_embd)
        cos_score = F.cosine_similarity(mu_last, x_hidden, dim=-1)  # (B, T)
        r = F.sigmoid(torch.div(cos_score, self.KT))

        return r, v_r_norm

    def forward(self, states, timesteps, actions=None, keys=None, future_states=None):
        B, T = states.shape[0], states.shape[1]
        states.requires_grad = True

        # Get state gradient
        r, v_r_norm = self.get_reward(states, keys)
        r_sum = r.sum()
        states_grad = torch.autograd.grad(r_sum, states, retain_graph=True, create_graph=True)[0]

        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T * 3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T * 3):3, :] = state_embeddings
        if keys is not None:
            # keys should not be None (???????)
            if self.use_future_state:
                assert future_states is not None
                goal_vecs = torch.cat([states_grad, future_states], dim=-1)
            else:
                goal_vecs = states_grad
            grad_embeddings = self.state_grad_encoder(goal_vecs)
            token_embeddings[:, 1:(T * 3):3, :] = grad_embeddings

        if actions is not None:
            # actions is None when at s0
            # the last action is not used as inputs during ActNet training.
            token_embeddings[:, 2:(T * 3 - 1):3, :] = self.action_encoder(actions[:, :(T - 1)])

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 3, dim=1)

        pos_emb = global_pos_emb + local_pos_emb
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, mid_feature = self.blocks(x)

        # Always predict next action
        x_action = self.ln(x)
        action_preds = self.action_predictor(x_action[:, 1:(T * 3):3, :])
        # Use it as ActNet, do action prediction
        return action_preds, r, v_r_norm


class ENet(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.attn_type == '-'  # only try to reconstruct key state back to state
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.block_size = config.block_size

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # key embeddings
        self.grad_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)
        # State predictor
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

    # (Must) include in skip_feature, to make sure the keys 'loss' some of the information from states
    def forward(self, state_grads, timesteps):
        B, T = state_grads.shape[0], state_grads.shape[1]
        key_embeddings = self.grad_encoder(state_grads)

        # Embeddings for key (action, and key state query) tokens.
        token_embeddings = key_embeddings

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 1, dim=1)
        pos_emb = global_pos_emb + local_pos_emb

        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        x = self.ln(x)
        eact_preds = self.action_predictor(x)

        return eact_preds


class RecNet(nn.Module):
    def __init__(self, config, state_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and key_dim > 0
        assert config.attn_type == '-'  # only try to reconstruct key state back to state
        self.config = config
        self.state_dim = state_dim
        self.key_dim = key_dim
        self.block_size = config.block_size

        # self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # key embeddings
        self.key_encoder = MLP(key_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)
        # State predictor
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[256, 256])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # (Must) include in skip_feature, to make sure the keys 'loss' some of the information from states
    def forward(self, keys, timesteps, skip_feature=None):
        B, T = keys.shape[0], keys.shape[1]
        key_embeddings = self.key_encoder(keys)

        # Embeddings for key (action, and key state query) tokens.
        token_embeddings = key_embeddings

        # # Set up position embeddings similar to that in Decision Transformer.
        # global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        # timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        # global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        # local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 1, dim=1)
        # pos_emb = global_pos_emb + local_pos_emb
        #
        # The keys are embedded with unified time steps !!!

        x = token_embeddings  # + pos_emb
        x = self.drop(x)
        x, _ = self.blocks(x, skip_feature)
        x = self.ln(x)
        state_preds = self.state_predictor(x)

        return state_preds


class ActCommitNet(nn.Module):
    """
    If using as ActNet:
        action prediction of a(t-1) based on s[0:(t-1)], k[0:(t-1)], a[0:(t-2)]
    If using as CommitNet:
        commit key state k(t-1) based on s[0:(t-1)], k[0:(t-2)], a[0:(t-2)]
    """
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        assert config.use_key_energy or key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.use_key_energy = config.use_key_energy

        self.block_size = config.block_size * 3

        self.commit = config.commit  # use it as CommitNet or ActNet, True will be CommitNet

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])
        if config.use_key_energy:
            self.key_encoder = MLP(key_dim + state_dim, config.n_embd, hidden_dims=[256])
        else:
            self.key_encoder = MLP(key_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # Action predictor
        if config.commit is True:
            # Use as CommitNet
            self.key_predictor = MLP(config.n_embd, key_dim, hidden_dims=[256, 256])
        else:
            # Use as ActNet
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

    def forward(self, states, timesteps, actions=None, keys=None, key_energy=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T*3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T*3):3, :] = state_embeddings
        if keys is not None:
            # key is None when at s0, committing, since keys are from KeyNet + KeyBook
            # keys should not be None (???????)
            if key_energy is not None:
                key_embeddings = torch.cat([keys, key_energy], dim=-1)
            else:
                key_embeddings = keys
            key_embeddings = self.key_encoder(key_embeddings)
            token_embeddings[:, 1:(T*3):3, :] = key_embeddings

        if actions is not None:
            # actions is None when at s0
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

        if self.commit is True:
            # Use it as CommitNet, do committing key prediction
            key_commit_soft = self.key_predictor(x[:, 0:(T*3):3, :])
            return key_commit_soft
        else:
            # Use it as ActNet, do action prediction
            act_preds = self.action_predictor(x[:, 1:(T*3):3, :])
            return act_preds


class FutureNet(nn.Module):
    """
    If using as ActNet:
        action prediction of a(t-1) based on s[0:(t-1)], k[0:(t-1)], a[0:(t-2)]
    If using as CommitNet:
        commit key state k(t-1) based on s[0:(t-1)], k[0:(t-2)], a[0:(t-2)]
    """
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim

        self.block_size = config.block_size * 3

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[256])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])
        self.key_encoder = MLP(key_dim, config.n_embd, hidden_dims=[256])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # state predictor
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[256, 256])

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


    def forward(self, states, timesteps, actions=None, keys=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T*3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T*3):3, :] = state_embeddings
        if keys is not None:
            # keys should not be None (???????)
            key_embeddings = self.key_encoder(keys)
            token_embeddings[:, 1:(T*3):3, :] = key_embeddings

        if actions is not None:
            # actions is None when at s0
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

        state_preds = self.state_predictor(x[:, 1:(T*3):3, :])
        return state_preds

