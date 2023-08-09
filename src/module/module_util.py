import torch
import torch.nn as nn


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
