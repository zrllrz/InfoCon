import torch
from torch import nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ConditionResnetBlockFC(nn.Module):
    def __init__(self, size_h=None, size_c=None):
        super().__init__()
        self.ln0 = nn.LayerNorm(size_h)
        self.ln0.bias.data.zero_()
        self.ln0.weight.data.fill_(1.0)
        self.res0 = ResnetBlockFC(size_h, size_h, size_h, beta=0)
        # condition part 0
        self.linear_c0 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.linear_c0.bias, 0.0)
        nn.init.kaiming_normal_(self.linear_c0.weight, a=0, mode="fan_in")

        self.ln1 = nn.LayerNorm(size_h)
        self.ln1.bias.data.zero_()
        self.ln1.weight.data.fill_(1.0)
        self.res1 = ResnetBlockFC(size_h, size_h, size_h, beta=0)
        # condition part 1
        self.linear_c1 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.linear_c1.bias, 0.0)
        nn.init.kaiming_normal_(self.linear_c1.weight, a=0, mode="fan_in")
        # ln_out for next...

    def forward(self, x, c):
        # x: (..., size_h)
        # c: (..., size_c)
        x = self.ln0(x + self.linear_c0(c))
        x = self.res0(x)
        x = self.ln1(x + self.linear_c1(c))
        x = self.res1(x)
        return x


class ImplicitSAResFC(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1):
        super().__init__()
        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim

        self.use_pos_emb = config.use_pos_emb
        if config.use_pos_emb:
            self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        # set it to be 128 first
        self.state_encoder = nn.Linear(state_dim, config.n_embd)
        nn.init.constant_(self.state_encoder.bias, 0.0)
        nn.init.kaiming_normal_(self.state_encoder.weight, a=0, mode="fan_in")

        self.key_encoder = nn.Linear(key_dim, config.n_embd)
        nn.init.constant_(self.key_encoder.bias, 0.0)
        nn.init.kaiming_normal_(self.key_encoder.weight, a=0, mode="fan_in")

        self.n_state_layer = config.n_state_layer
        self.n_action_layer = config.n_action_layer

        # first predict out state
        self.state_block = nn.ModuleList(
            [ConditionResnetBlockFC(size_h=config.n_embd, size_c=config.n_embd) for _ in range(config.n_state_layer)]
        )

        self.ln_state = nn.LayerNorm(config.n_embd)
        self.ln_state.bias.data.zero_()
        self.ln_state.weight.data.fill_(1.0)

        self.state_predictor = nn.Linear(config.n_embd, state_dim)
        nn.init.constant_(self.state_predictor.bias, 0.0)
        nn.init.kaiming_normal_(self.state_predictor.weight, a=0, mode="fan_in")

        # then predict action
        self.action_block = nn.ModuleList(
            [ConditionResnetBlockFC(size_h=config.n_embd, size_c=config.n_embd) for _ in range(config.n_action_layer)]
        )

        self.ln_action = nn.LayerNorm(config.n_embd)
        self.ln_action.bias.data.zero_()
        self.ln_action.weight.data.fill_(1.0)

        self.action_predictor = nn.Linear(config.n_embd, action_dim)
        nn.init.constant_(self.action_predictor.bias, 0.0)
        nn.init.kaiming_normal_(self.action_predictor.weight, a=0, mode="fan_in")

    def forward(self, states, timesteps, keys=None, predict_state=False):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        c_key = self.key_encoder(keys)

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 1, dim=1)
        pos_emb = global_pos_emb + local_pos_emb

        x_state = state_embeddings + pos_emb  # selectable
        c_key = c_key + pos_emb  # selectable

        # inference next state
        h_next_state = x_state
        for block in self.state_block:
            h_next_state = block(h_next_state, c_key)
        if predict_state:
            state_preds = self.state_predictor(self.ln_state(h_next_state))
        else:
            state_preds = None

        # inference corresponding action
        h_action = x_state
        for block in self.action_block:
            h_action = block(h_action, h_next_state)
        action_preds = self.action_predictor(self.ln_action(h_action))

        return action_preds, state_preds


class ExplicitSAHN(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, e_dim=-1):
        super().__init__()
        assert state_dim > 0 and action_dim > 0 and e_dim > 0
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.e_dim = e_dim

        self.dim_h = config.dim_h

        self.reward_layer = config.reward_layer  # numbers of layers in reward net
        # when it is n, it is like
        # s ->[in layer (ds+1)*h] hs0 ->[l0 h*(h+1)]->[l1 h*(h+1)]->...->[l(n-2) h*(h+1)]->[l(n-1) h]
        # last layer has no bias

        # HYPER PART
        # first layers:  config.reward_layer x [ds -> dz, dz == config.reward_layer]
        assert e_dim % config.reward_layer == 0
        self.e_dim_slice = e_dim // config.reward_layer
        self.hyper_block_group = nn.ModuleList()
        for i in range(config.reward_layer):
            self.hyper_block_group.append(
                nn.Sequential(
                    nn.Linear(e_dim // config.reward_layer, e_dim // config.reward_layer),
                    nn.SiLU(),
                    nn.Linear(e_dim // config.reward_layer, config.reward_layer)
                )
            )
            nn.init.constant_(self.hyper_block_group[i][0].bias, 0.0)
            nn.init.kaiming_normal_(self.hyper_block_group[i][0].weight, a=0, mode="fan_in")
            nn.init.constant_(self.hyper_block_group[i][2].bias, 0.0)
            nn.init.kaiming_normal_(self.hyper_block_group[i][2].weight, a=0, mode="fan_in")


        # second layers: if hidden layers, h * (h + 1) params, weighted from config.reward_layer choice
        # if last layer, h params,
        self.hyper_block_hidden_out = nn.Linear(config.reward_layer, config.dim_h * (config.dim_h + 1), bias=False)
        nn.init.kaiming_normal_(self.hyper_block_hidden_out.weight, a=0, mode="fan_in")

        self.hyper_block_score_out = nn.Linear(config.reward_layer, config.dim_h, bias=False)
        nn.init.kaiming_normal_(self.hyper_block_score_out.weight, a=0, mode="fan_in")

        self.use_pos_emb = config.use_pos_emb
        if config.use_pos_emb:
            self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.dim_h))
            self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.dim_h))

        # State embeddings & Action embeddings & Key embeddings
        # set it to be 128 first
        self.state_encoder = nn.Linear(state_dim, config.dim_h)
        nn.init.constant_(self.state_encoder.bias, 0.0)
        nn.init.kaiming_normal_(self.state_encoder.weight, a=0, mode="fan_in")

        self.action_predictor = nn.Linear(config.dim_h, action_dim)
        nn.init.constant_(self.action_predictor.bias, 0.0)
        nn.init.kaiming_normal_(self.action_predictor.weight, a=0, mode="fan_in")

        # Policy Using Gradient
        # currently do not usse context like transformer style, only includes in states

        self.pl0 = nn.Linear(config.dim_h, config.dim_h)
        self.pln0 = nn.LayerNorm(config.dim_h)
        self.pa0 = nn.SiLU()
        self.pc0 = nn.Linear(state_dim, config.dim_h)

        self.pl1 = nn.Linear(config.dim_h, config.dim_h)
        self.pln1 = nn.LayerNorm(config.dim_h)
        self.pa1 = nn.SiLU()
        self.pc1 = nn.Linear(state_dim, config.dim_h)

        self.pl2 = nn.Linear(config.dim_h, config.dim_h)
        self.pln2 = nn.LayerNorm(config.dim_h)
        self.pa2 = nn.SiLU()
        self.pc2 = nn.Linear(state_dim, config.dim_h)

        self.pl3 = nn.Linear(config.dim_h, config.dim_h)
        self.pln3 = nn.LayerNorm(config.dim_h)
        self.pa3 = nn.SiLU()
        self.pc3 = nn.Linear(state_dim, config.dim_h)

    def get_state_grad(self, states, timesteps, keys=None):
        states.requires_grad = True
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)  # (B, T, dim_h)

        # Set up position embeddings similar to that in Decision Transformer.
        global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
        timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.dim_h, dim=-1)
        global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 1, dim=1)
        pos_emb = global_pos_emb + local_pos_emb
        x_states = state_embeddings + pos_emb  # (B, T, h)

        # hypernet part generate parameters then feed forward...
        for i in range(self.reward_layer - 1):
            print('keys.shape =', keys.shape)
            key_i = keys[:, :, i*self.e_dim_slice:(i+1)*self.e_dim_slice]  # (B, T, e_dim//reward_layer)
            # print(self.hyper_block_group[i].device)
            print('key_i.shape =', key_i.shape)
            weight = self.hyper_block_group[i](key_i)  # (B, T, reward_layer)
            wb_i = self.hyper_block_hidden_out(weight)  # (B, T, h*(h+1))
            w_i = wb_i[..., :self.dim_h*self.dim_h].view(B, T, self.dim_h, self.dim_h)  # (B, T, h, h)
            b_i = wb_i[..., (-self.dim_h):]  # (B, T, h)
            x_states = b_i + torch.matmul(x_states.view(B, T, 1, self.dim_h), w_i).squeeze(2)
        key_last = keys[:, :, (self.reward_layer-1)*self.e_dim_slice:]
        weight_last = self.hyper_block_group[self.reward_layer-1](key_last)  # (B, T, reward_layer)
        w_last = self.hyper_block_hidden_out(weight_last)  # (B, T, h)
        reward = torch.matmul(x_states.view(B, T, self.dim_h), w_last.view(B, T, self.dim_h, 1)).squeeze(2)  # (B, T)
        print('reward.shape =', reward.shape)

        # get the gradient of states
        state_grads = torch.autograd.grad(reward, states, retain_graph=True, create_graph=True)
        return state_grads, x_states

    def policy_with_grads(self, state_grads, state_embeddings):

        x = state_embeddings
        h = self.pa0(self.pln0(self.pl0(x) * self.pc0(state_grads)))
        x = self.pa1(x + self.pln1(self.pl1(h) * self.pc1(state_grads)))

        h = self.pa2(self.pln2(self.pl2(x) * self.pc2(state_grads)))
        x = self.pa3(x + self.pln3(self.pl3(h) * self.pc3(state_grads)))

        return self.action_predictor(x)

    def forward(self, states, timesteps, keys=None):
        state_grads, x_states = self.get_state_grad(states, timesteps, keys)
        action_preds = self.policy_with_grads(state_grads, x_states)

        return action_preds
