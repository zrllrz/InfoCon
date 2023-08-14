"""
Reference:
https://github.com/ashawkey/stable-dreamfusion
"""

import torch
import torch.nn as nn
from math import log


class FreqEncoder(nn.Module):
    def __init__(self, length=32):
        super().__init__()

        self.length = length
        coe_freq = torch.remainder(torch.arange(length), 32)
        coe_freq = torch.mul(torch.pow(2.0, coe_freq), torch.pi)
        self.register_buffer('coe_freq', coe_freq.unsqueeze(0))

    def forward(self, unified_t):
        u_t = torch.mul(torch.sub(torch.mul(unified_t, 2.0), 1.0), torch.pi).unsqueeze(-1)
        emb_cos = torch.cos(u_t @ self.coe_freq)
        emb_sin = torch.sin(u_t @ self.coe_freq)
        emb_t = torch.cat([emb_cos, emb_sin], dim=-1)
        return emb_t


class TimeSphereEncoder(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate * 2.0

    def forward(self, feature, unified_t):
        # feature: (..., d)
        # unified_t: (...)
        u_t = torch.mul(torch.sub(torch.mul(unified_t, 2.0), 1.0), torch.pi).unsqueeze(-1)  # (..., 1)
        u_t_sin = torch.sin(torch.div(u_t, self.rate))  # (..., 1)
        u_t_cos = torch.cos(torch.div(u_t, self.rate))  # (..., 1)
        f_cos = u_t_cos * feature  # (..., d)
        f_t = torch.cat([u_t_sin, f_cos], dim=-1)
        return f_t


class mereNLL(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
        self.one_minus_eps = 1.0 - eps
        self.log_one_minus_eps = log(1.0 - eps)

    def forward(self, x):
        one_minus_eps_mul_x = x * self.one_minus_eps
        coe = self.eps + one_minus_eps_mul_x
        v_log = coe * torch.log(torch.div(coe, one_minus_eps_mul_x))
        return v_log + self.log_one_minus_eps


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


# Resnet Blocks
class CondResnetBlockFC(nn.Module):
    """
    Fully connected Conditional ResNet Block class.
    :param size_h (int): hidden dimension
    :param size_c (int): latent dimension
    """

    def __init__(self, size_h, size_c, beta=0):
        super().__init__()

        # Main Branch
        self.fc_0 = nn.Linear(size_h, size_h)
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")

        self.ln_0 = nn.LayerNorm(size_h)
        self.ln_0.bias.data.zero_()
        self.ln_0.weight.data.fill_(1.0)

        self.fc_1 = nn.Linear(size_h, size_h)
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        self.ln_1 = nn.LayerNorm(size_h)
        self.ln_1.bias.data.zero_()
        self.ln_1.weight.data.fill_(1.0)

        # Conditional Branch
        self.c_fc_0 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.c_fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.c_fc_0.weight, a=0, mode="fan_in")

        self.c_fc_1 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.c_fc_1.bias, 0.0)
        nn.init.kaiming_normal_(self.c_fc_1.weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, x, c, last_activation=True):
        h = self.fc_0(x)
        h = self.ln_0(h * self.c_fc_0(c))
        h = self.activation(h)

        h = self.fc_1(h)
        h = self.ln_1(h * self.c_fc_1(c))

        out = x + h
        if last_activation:
            out = self.activation(out)

        return out
