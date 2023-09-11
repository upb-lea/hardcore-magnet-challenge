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
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, 
                 residual=True, double_layered=True, dropout=0.2, act_func=None):
        super(TemporalBlock, self).__init__()
        padding = ((kernel_size-1) // 2) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                                    padding=padding, dilation=dilation, 
                                                    padding_mode='circular'))
        self.relu1 = ACTIVATION_FUNCS.get(act_func, nn.Identity)()
        self.dropout1 = nn.Dropout1d(dropout)
        if double_layered:
            self.relu2 = nn.Identity()
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation, 
                                                        padding_mode='circular'))
            self.dropout2 = nn.Dropout1d(dropout)
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                     self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()
        self.residual = residual
        if residual:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        else:
            self.downsample = None
        self.double_layered = double_layered
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.double_layered:
            self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            y = torch.clip(out + res, -10, 10)  # out += res is not allowed, it would become an inplace op, weird
            y = self.relu(y)
        else:
            y = out
        return y
    
class TemporalAcausalConvNet(nn.Module):
    def __init__(self, num_inputs, layer_cfg=None):
        super().__init__()
        layer_cfg = layer_cfg or {'f': [{'units': 32, 'act_func': 'tanh'}, #{'units': 6, 'act_func': 'tanh'}, 
                                        {'units': 1}]}
        layers = []
        dilation_offset = layer_cfg.get("starting_dilation_rate", 0)  # >= 0
        for i, l_cfg in enumerate(layer_cfg['f']):
            kernel_size = l_cfg.get('kernel_size', 7)
            dropout_rate = layer_cfg.get("dropout", 0.0)
            dilation_size = 2 ** (i + dilation_offset)
            in_channels = num_inputs if i == 0 else layer_cfg['f'][i-1]['units']
            layers += [TemporalBlock(in_channels, l_cfg['units'], kernel_size, stride=1, dilation=dilation_size,
                                     residual=layer_cfg.get('residual', False),
                                     double_layered=layer_cfg.get("double_layered", False),
                                     dropout=dropout_rate, act_func=l_cfg.get('act_func', nn.Identity)),
                       ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y = self.network(x)
        # subtract mean across time domain
        y = y - y.mean(dim=-1).unsqueeze(-1)
        return y

class TCNWithScalarsAsBias(nn.Module):
    def __init__(self, num_input_scalars, num_input_ts=1, tcn_layer_cfg=None, scalar_layer_cfg=None):
        super().__init__()
        self.num_input_ts = num_input_ts
        self.x_idx_for_ts = [-a for a in range(1, num_input_ts+1)]
        tcn_layer_cfg = tcn_layer_cfg or {'f': [{'units': num_input_scalars+num_input_ts, 'act_func': 'tanh'}, 
                                                {'units': 32, 'act_func': 'tanh'}, 
                                                {'units': 1}]} 
        scalar_layer_cfg = scalar_layer_cfg or {'f': [{'units': num_input_scalars, 'act_func': 'tanh'}]}  # only one layer
        tcn_layers = []
        dilation_offset = tcn_layer_cfg.get("starting_dilation_rate", 0)  # >= 0
        for i, l_cfg in enumerate(tcn_layer_cfg['f']):
            kernel_size = l_cfg.get('kernel_size', 9)
            dropout_rate = tcn_layer_cfg.get("dropout", 0.0)
            dilation_size = 2 ** (i + dilation_offset)
            if i == 0:
                in_channels = num_input_ts  # TCN acts only on B field(s) in first layer
            else:
                in_channels = tcn_layer_cfg['f'][i-1]['units']  
            tcn_layers += [TemporalBlock(in_channels, l_cfg['units'], kernel_size, stride=1, dilation=dilation_size,
                                     residual=tcn_layer_cfg.get('residual', False),
                                     double_layered=tcn_layer_cfg.get("double_layered", False),
                                     dropout=dropout_rate, act_func=l_cfg.get('act_func', nn.Identity)),
                       ]
            if i == 0:
                self.b_proc_layer = tcn_layers.pop()
        self.upper_tcn = nn.Sequential(*tcn_layers)
        self.scalar_layer = TemporalBlock(num_input_scalars, num_input_scalars,  kernel_size=1, stride=1, dilation=1, residual=False, double_layered=False,
                                     dropout=False, act_func=scalar_layer_cfg['f'][0]['act_func'])
    
    def forward(self, x):
        b_proc = self.b_proc_layer(x[:, self.x_idx_for_ts, :])
        scalar_proc = self.scalar_layer(x[:, :-self.num_input_ts, :])
        catted = torch.cat([b_proc[:, :-scalar_proc.shape[1], :], b_proc[:, -scalar_proc.shape[1]:, :] + scalar_proc], dim=1)
        y = self.upper_tcn(catted)
        y = y + x[:, [-self.num_input_ts], :]  # residual connection to per-profile scaled B curve

        # subtract mean across time domain
        y = y - y.mean(dim=-1).unsqueeze(-1)

        return y


