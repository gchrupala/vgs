import torch.nn as nn
import torch.nn.functional as F
import onion.util as util
import torch

class SelfAttention(nn.Module):
    """Parameterized weighted average of a sequence of states."""
    def __init__(self, size_in, size, activation=F.tanh):
        super(SelfAttention, self).__init__()
        util.autoassign(locals())
        self.Regress1 = util.make_linear(self.size_in, self.size)
        self.Regress2 = util.make_linear(self.size, 1)

    def forward(self, h):
        alpha = softmax_time(self.Regress2(self.activation(self.Regress1(h))))
        return (alpha.expand_as(h) * h).sum(dim=1)


def softmax_time(x):
    """Input has shape Batch x Time x 1. Return softmax over dimension Time."""
    B, T, _ = x.size()
    return F.softmax(x.view((B, T))).view((B, T, 1))
