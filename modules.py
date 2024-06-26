import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math

import math

import torch
from torch import Tensor, nn


# Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding
# from https://github.com/JHLew/Learnable-Fourier-Features/blob/main/positional_encoding.py
class LearnableFourierFeatures(nn.Module):
    def __init__(self, pos_dim: int, f_dim: int, h_dim=0, d_dim=0, gamma=1.0):
        """
        Implements learned fourier feature positional encodings

        Args:
            pos_dim (int): dimensionality of positions
            f_dim (int): dim of fourier features
            h_dim (int): hidden dim of mlp
            d_dim (int): output dim
            gamma (float, optional): scaling factor for intialization. Defaults to 1.0.
                corresponds to length scale to use of Gaussian kernel approx
        """
        super(LearnableFourierFeatures, self).__init__()
        assert (
            f_dim % 2 == 0
        ), "number of fourier feature dimensions must be divisible by 2."
        enc_f_dim = int(f_dim / 2)
        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) / gamma**2)
        # self.mlp = nn.Sequential(
        #     nn.Linear(f_dim, h_dim), nn.GELU(), nn.Linear(h_dim, d_dim)
        # )
        self.div_term = math.sqrt(f_dim)

    def forward(self, pos: Tensor) -> Tensor:
        """
        Encods pos into embedding

        Args:
            pos (Tensor): shape [..., M]

        Returns:
            Tensor: shape [..., D]
        """
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term

        pos_enc = F  # self.mlp(F)

        return pos_enc
    
class RELUEmbeddingFunction(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_features=512, num_hidden_layers=3):
        super(RELUEmbeddingFunction, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers

        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_features))
        self.layers.append(nn.ReLU())
        for i in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_features, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, input_transform_function=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        self.input_transform_function = input_transform_function
        # print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        if self.input_transform_function is None:
            coords = coords_org
        else:
            coords = self.input_transform_function(coords_org)
        # import pdb;pdb.set_trace()
        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

class SingleBVPNet_nd(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, input_transform_function=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        self.input_transform_function = input_transform_function
        # print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        if self.input_transform_function is None:
            coords = coords_org
        else:
            coords = self.input_transform_function(coords_org)
        # import pdb;pdb.set_trace()
        output = self.net(coords)

        # implementing the requirements of the 2nd order upwind scheme
        delta_x = torch.from_numpy(np.array([0.00008, 0.5,1.5,0.01])).float().cuda()
        du_nd =  torch.zeros_like(coords_org)
        for i,delta_x_i in enumerate(delta_x):
            delta_x_i_full = torch.zeros_like(delta_x)
            delta_x_i_full[i] = delta_x_i
            x_minus = model_input['coords'] - delta_x
            x_minus_times_two = x_minus - delta_x
            model_out_delta =  self.net(x_minus)
            model_out_two_delta =  self.net(x_minus_times_two)
            du_nd[...,i:i+1] = 0.5*(3*output - 4*model_out_delta + model_out_two_delta)/delta_x_i


        return {'model_in': coords_org, 'model_out': output, 
                'du_nd': du_nd}

class CompositeModel(torch.nn.Module):
    def __init__(self, embedding_function, deepreach_model):
        super(CompositeModel, self).__init__()
        self.embedding_function = embedding_function
        self.deepreach_model = deepreach_model

    def forward(self, x):
        
        x = self.embedding_function(x)
        dr_input = {"coords": x}
        x = self.deepreach_model(dr_input)["model_out"]
        return x


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
