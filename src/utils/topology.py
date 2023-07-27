import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class Biased_Elu(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1


class SinusAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class GeneralizedCosinusUnit(nn.Module):
    def forward(self, x):
        return torch.cos(x)*x


ACTIVATION_FUNCS = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU,
                    'biased_elu': Biased_Elu, 'sinus': SinusAct, 'gcu': GeneralizedCosinusUnit}


class DifferenceEqLayer(nn.Module):
    """For discretized ODEs"""

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        device = torch.device('cpu')
        self.cell = cell(*cell_args, **cell_kwargs).to(device)

    def forward(self, input, state):
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

class ExplEulerCell(nn.Module):
    def __init__(self, n_targets, n_input_feats, subsample_factor=1,
                 layer_cfg=None):
        super().__init__()

        self.output_size = n_targets
        self.subsample_factor = subsample_factor

        # layer config init
        layer_default = {
            'f': [{'units': 32, 'activation': 'relu'},
                  {'units': n_targets, 'activation': 'sinus'}],
        }
        self.layer_cfg = layer_cfg or layer_default

        # main sub NN
        f_layers = []
        f_units = n_input_feats + n_targets

        for layer_specs in self.layer_cfg['f']:
            lay = nn.Linear(f_units, layer_specs["units"])
            lay.weight.data.normal_(0, 1e-2)
            lay.bias.data.normal_(0, 1e-2)
            f_layers.append(lay)
            if layer_specs.get('activation', 'linear') != 'linear':
                f_layers.append(ACTIVATION_FUNCS[layer_specs['activation']]())
            f_units = layer_specs["units"]
        self.f = nn.Sequential(*f_layers)


    def forward(self, inp, hidden):
        prev_out = hidden
        freq = inp[:, [0]]  # by convention first input feature
        norm_freq = freq / 500_000  # Hz
        all_input = torch.cat([norm_freq, inp[:, 1:], hidden], dim=1)
        out = prev_out + self.subsample_factor/(freq*1024) * self.f(all_input)
        return prev_out, out