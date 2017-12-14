import torch.nn as nn
import torch
import onion.util as util
import numbers
import numpy as np
import torch.nn.functional as F
from functools import reduce

floatX = 'float32'

class Linear(nn.Module):

    def __init__(self, in_size, out_size, bias_init=None, init_scale=0.04):
        super(Linear, self).__init__()
        util.autoassign(locals())
        self.w = torch.nn.Parameter(self.make_param((self.in_size, self.out_size), 'uniform'))
        if bias_init is not None:
            self.b = torch.nn.Parameter(self.make_param((self.out_size,), self.bias_init))

    def make_param(self, shape, init_scheme):
        """Create variables which are used as trainable model parameters."""
        if isinstance(init_scheme, numbers.Number):
            init_value = np.full(shape, init_scheme, floatX)
        elif init_scheme == 'uniform':
            #init_value = self._np_rng.uniform(low=-self.init_scale, high=self.init_scale, size=shape).astype(floatX) # FIXME
            init_value = np.random.uniform(low=-self.init_scale, high=self.init_scale, size=shape).astype(floatX)

        else:
            raise AssertionError('unsupported init_scheme')
        p = torch.from_numpy(init_value)
        return p

    def forward(self, x):
        if self.bias_init is not None:
            return torch.matmul(x, self.w) + self.b
        else:
            return torch.matmul(x, self.w)

class RHN(nn.Module):
    """Recurrent Highway Network. Based on
    https://arxiv.org/abs/1607.03474 and
    https://github.com/julian121266/RecurrentHighwayNetworks.

    """
    def __init__(self, size_in, size, recur_depth=1, drop_i=0.75 , drop_s=0.25,
                 init_T_bias=-2.0, init_H_bias='uniform', tied_noise=True, init_scale=0.04, seed=1):
        super(RHN, self).__init__()
        util.autoassign(locals())
        hidden_size = self.size
        self.LinearH = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_H_bias)
        self.LinearT = Linear(in_size=self.size_in, out_size=hidden_size, bias_init=self.init_T_bias)
        self.recurH = nn.ModuleList()
        self.recurT = nn.ModuleList()
        for l in range(self.recur_depth):
            if l == 0:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size))
            else:
                self.recurH.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_H_bias))
                self.recurT.append(Linear(in_size=hidden_size, out_size=hidden_size, bias_init=self.init_T_bias))

    def apply_dropout(self, x, noise):
        if self.training:
            return noise * x
        else:
            return x

    def get_dropout_noise(self, shape, dropout_p):
        keep_p = 1 - dropout_p
        noise = (1. / keep_p) * torch.bernoulli(torch.zeros(shape) + keep_p)
        if torch.cuda.is_available():
            noise = noise.cuda()
        noise = torch.autograd.Variable(noise)
        return noise

    def step(self, i_for_H_t, i_for_T_t, h_tm1, noise_s):
        tanh, sigm = F.tanh, F.sigmoid
        noise_s_for_H = noise_s if self.tied_noise else noise_s[0]
        noise_s_for_T = noise_s if self.tied_noise else noise_s[1]

        hidden_size = self.size
        s_lm1 = h_tm1
        for l in range(self.recur_depth):
            s_lm1_for_H = self.apply_dropout(s_lm1, noise_s_for_H)
            s_lm1_for_T = self.apply_dropout(s_lm1, noise_s_for_T)
            if l == 0:
                # On the first micro-timestep of each timestep we already have bias
                # terms summed into i_for_H_t and into i_for_T_t.
                H = tanh(i_for_H_t + self.recurH[l](s_lm1_for_H))
                T = sigm(i_for_T_t + self.recurT[l](s_lm1_for_T))
            else:
                H = tanh(self.recurH[l](s_lm1_for_H))
                T = sigm(self.recurT[l](s_lm1_for_T))
            s_l = (H - s_lm1) * T + s_lm1
            s_lm1 = s_l

        y_t = s_l
        return y_t

    def forward(self, h0, seq, repeat_h0=1):
        #print(seq.size())
        inputs = seq.permute(1, 0, 2)
        (_seq_size, batch_size, _) = inputs.size()
        hidden_size = self.size
        # We first compute the linear transformation of the inputs over all timesteps.
        # This is done outside of scan() in order to speed up computation.
        # The result is then fed into scan()'s step function, one timestep at a time.
        noise_i_for_H = self.get_dropout_noise((batch_size, self.size_in), self.drop_i)
        noise_i_for_T = self.get_dropout_noise((batch_size, self.size_in), self.drop_i) if not self.tied_noise else noise_i_for_H

        i_for_H = self.apply_dropout(inputs, noise_i_for_H)
        i_for_T = self.apply_dropout(inputs, noise_i_for_T)

        i_for_H = self.LinearH(i_for_H)
        i_for_T = self.LinearT(i_for_T)


        # Dropout noise for recurrent hidden state.
        noise_s = self.get_dropout_noise((batch_size, hidden_size), self.drop_s)
        if not self.tied_noise:
          noise_s = torch.stack(noise_s, self.get_dropout_noise((batch_size, hidden_size), self.drop_s))

        H0 = h0.expand((batch_size, self.size)) if repeat_h0 else h0
        #out, _ = theano.scan(self.step,
        #                     sequences=[i_for_H, i_for_T],
        #                     outputs_info=[H0],
        #                     non_sequences = [noise_s])
        out = []

        for t in range(len(i_for_H)):
            out.append(self.step(i_for_H[t], i_for_T[t], H0, noise_s))
        return torch.stack(out).permute(1, 0, 2)


class RHNH0(nn.Module):
    def __init__(self, size_in, size, fixed=False, **kwargs):
        """An RHN layer with its own initial state."""
        super(RHNH0, self).__init__()
        util.autoassign(locals())
        self.h0 = FixedZeros(self.size) if self.fixed else Zeros(self.size)
        self.RHN = RHN(size_in, size, **kwargs)

    def forward(self, inp):
        return self.RHN(self.h0(), inp)

class WithH0(nn.Module):
    def __init__(self, RNN, fixed=False):
        """An recurrent layer with its own initial state."""
        super(WithH0, self).__init__()
        self.RNN = RNN
        if fixed:
            self.h0 = FixedZeros(self.RNN.size)
        else:
            self.h0 = Zeros(self.RNN.size)

    def forward(self, inp):
        return self.RNN(self.h0(), inp)

class Zeros(nn.Module):
    """Returns a Parameter vector of specified size initialized with zeros."""
    def __init__(self, size):
        super(Zeros, self).__init__()
        util.autoassign(locals())
        self.zeros = torch.nn.Parameter(torch.zeros(self.size))

    def forward(self):
        return self.zeros

class FixedZeros(nn.Module):
    """Returns a vector of specified size initialized with zeros."""
    def __init__(self, size):
        super(FixedZeros, self).__init__()
        util.autoassign(locals())
        self.zeros = torch.autograd.Variable(torch.zeros(self.size), requires_grad=True)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()

    def forward(self):
        return self.zeros

class Residual(nn.Module):
    """Residualizes a layer."""
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, inp):
        return inp + self.layer(inp)

class Identity(nn.Module):
    """Identity layer."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inp):
        return inp

class Compose(nn.Module):

    def __init__(self, first, second):
        super(Compose, self).__init__()
        self.first = first
        self.second = second

    def forward(self, inp):
        return self.first(self.second(inp))

    def intermediate(self, inp):
        x = self.second(inp)
        z = self.first(x)
        return (x,z)

class StackedRHN(nn.Module):
    def __init__(self, size_in, size, depth=2, residual=False, fixed=False, **kwargs):
        super(StackedRHN, self).__init__()
        util.autoassign(locals())
        f = lambda x: Residual(x) if self.residual else x
        self.layers = torch.nn.ModuleList(
            [ f(RHNH0(self.size, self.size, fixed=self.fixed, **self.kwargs))  for _ in range(1,self.depth) ] )
        self.bottom = RHN(self.size_in, self.size, **self.kwargs)
        self.stack = reduce(lambda z, x: Compose(x, z), self.layers, Identity())

    def forward(self, h0, inp, repeat_h0=0):
        return self.stack(self.bottom(h0, inp, repeat_h0=repeat_h0))

    def intermediate(self, h0, inp, repeat_h0=0):
        zs = [ self.bottom(h0, inp, repeat_h0=repeat_h0) ]
        for layer in self.layers:
            z = layer(zs[-1])
            zs.append(z)

        return torch.stack(zs).permute(1,2,0,3)

class StackedRHNH0(nn.Module):
    "Stack of RHNs with own initial state."

    def __init__(self, size_in, size, depth=2, residual=False, fixed=False, **kwargs):
        super(StackedRHNH0, self).__init__()
        util.autoassign(locals())
        self.layer = WithH0(StackedRHN(size_in, size, depth=depth, residual=residual, **kwargs), fixed=fixed)

    def forward(self, inp):
        return self.layer(inp)

    def intermediate(self, inp):
        return self.layer.intermediate(self.layer.h0(), inp)
