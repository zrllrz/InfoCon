"""
Code for the model architecture of CoTPC, based on the GPT implementation.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


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


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(self, block_size, **kwargs):
        assert kwargs['model_type'] in ['s', 's+a', 's+k', 's+a+k'], \
            f"Unsupported model_type: {kwargs['model_type']}"

        if '+a' in kwargs['model_type']:  # If the action history is used.
            self.block_size = block_size * 2
        else:
            self.block_size = block_size

        if 'k' in kwargs['model_type']:
            self.use_key_states = True
        else:
            self.use_key_states = False

            # It is in the form of e.g., '023', meaning the features out of attention
            # layers of idx 0, 2, 3 are used for key state prediction losses.
            # assert kwargs['key_state_loss'] not in ['', None] and \
            #     np.all([l.isnumeric() for l in kwargs['key_state_loss']])
            #
            # self.key_state_loss = kwargs['key_state_loss']

        # Set up other attributes.
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttentionWithCoT(nn.Module):
    """
    A multi-head masked self-attention layer equipped with key state query tokens for
    chain-of-thought predictive control. It is adapted from the minGPT repo.
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
        block_size = config.block_size
        self.register_buffer("mask",
                             torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        self.n_head = config.n_head
        self.model_type = config.model_type

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Masked attention

        # Masks used for the learnable key state query tokens, which are not causal (auto-regressive).

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    A Transformer block with masks specified for the learnable key state query tokens.
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithCoT(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, key_state_mask=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BlocksWithCoT(nn.Module):
    """
    A wrapper class for a sequence of Transformer blocks with masks specified for
    the learnable key state query tokens.
    """

    def __init__(self, config):
        super().__init__()
        # Register all the individual blocks.
        self.block_list = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.model_type = config.model_type
        self.n_head = config.n_head

    def forward(self, x):
        B, T, _ = x.shape

        output = []  # Also keep the intermediate results.
        for block in self.block_list:
            x = block(x)
            output.append(x)

        return x, output


class GPTCat(nn.Module):
    """
    GPT implementation with the support of the learnable key state query tokens,
    which is used for the chain-of-thought predictive control. Here, the context size
    is specified as block_size, which does not count the key state query tokens.
    """

    def __init__(self, config, state_dim=-1, action_dim=-1):
        super().__init__()

        assert state_dim > 0 and action_dim > 0
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = config.model_type
        self.block_size = config.block_size
        self.use_key_states = config.use_key_states

        # Set up learnable position embedding synchronized for s and a tokens, as proposed
        # in Decision Transformer. We use a similar global+local position embedding design.
        p_size = config.block_size // 2 if '+a' in self.model_type else config.block_size
        self.local_pos_emb = nn.Parameter(torch.zeros(1, p_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, config.max_timestep, config.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers) with CoT.
        self.blocks = BlocksWithCoT(config)

        # State embeddings.
        self.state_encoder = MLP(self.state_dim, config.n_embd, hidden_dims=[256])

        # Action embeddings.
        if '+a' in self.model_type:
            if self.use_key_states:
                self.action_encoder = MLP(action_dim + state_dim, config.n_embd, hidden_dims=[256])
            else:
                self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[256])

        # Action predictor.
        self.ln = nn.LayerNorm(config.n_embd)
        if self.use_key_states:
            self.action_predictor = MLP(config.n_embd, action_dim + state_dim, hidden_dims=[256, 256])
        else:
            self.action_predictor = MLP(config.n_embd, action_dim, hidden_dims=[256, 256])

        # Key state predictors. By default, we only use one predictor which takes
        # features from one attention layer.
        # if 'cot' in self.model_type:
        #     key_state_predictors = []
        #     for _ in self.key_state_loss:
        #         key_state_predictor = MLP(
        #             config.n_embd, self.state_dim, hidden_dims=[int(self.cot_decoder)])
        #         key_state_predictors.append(key_state_predictor)
        #     # Register all the key state predictors.
        #     self.key_state_predictors = nn.ModuleList(key_state_predictors)

        self.apply(self._init_weights)
        print(f"Total # of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Given state (and action) history, predict actions (and key states as CoT).
    # `timesteps` is used for the global+local position embedding design similar
    # to the one in Decision Transformer. `key_state_mask` is used so that the
    # (all-to-all) key state query tokens can attend to later tokens.
    def forward(self, states, timesteps, actions=None, key_states=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)

        # Embeddings for state (action, and key state query) tokens.
        token_embeddings = torch.zeros([B, self.block_size, self.config.n_embd],
                                       dtype=torch.float32, device=states.device)

        # If using action history as inputs: during training, all actions are
        # specified; during inference, only actions in the past are specified.
        # That is, the first action prediction has no action history as inputs.
        if '+a' in self.model_type:
            token_embeddings[:, :T*2:2, :] = state_embeddings
            if actions is not None:
                # Assume the last action is not used as inputs during training.
                if key_states is not None:
                    aks = torch.cat([actions[:, :T-1], key_states[:, :T-1]], dim=-1)
                else:
                    aks = actions[:, :T-1]
                token_embeddings[:, 1:T*2-1:2, :] = self.action_encoder(aks)

        else:
            token_embeddings[:, :T, :] = state_embeddings

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(
            global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb, 2, dim=1) \
            if '+a' in self.model_type else self.local_pos_emb

        x = token_embeddings + global_pos_emb + local_pos_emb

        x = self.drop(x)
        x, intermediate_feats = self.blocks(x)
        x = self.ln(x)
        ak_preds = self.action_predictor(x)

        # Get rid of dims for action tokens.
        if '+a' in self.model_type:
            # Remove the extra tokens when in eval mode.
            ak_preds = ak_preds[:, :T*2:2, :]

        if self.use_key_states:
            a_preds = ak_preds[:, :, :self.action_dim]
            k_preds = ak_preds[:, :, self.action_dim:]
        else:
            a_preds = ak_preds[:, :, :self.action_dim]
            k_preds = None

        return a_preds, k_preds

    def configure_adamw_optimizers(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
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
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('local_pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
                % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config['init_lr'],
            betas=(config['beta1'], config['beta2'])
        )
        return optimizer
