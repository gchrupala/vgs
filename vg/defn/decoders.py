import torch
import torch.nn as nn
import onion.util as util
import torch.nn.functional as F

# BAHDANAU-STYLE ATTENTION

class AlignAttention(nn.Module):
    """Soft alignment between two sequences, aka attention. Based on https://arxiv.org/abs/1508.04025v5."""

    def __init__(self, size_in):
        super(AlignAttention, self).__init__()
        util.autoassign(locals())
        self.W = nn.Linear(self.size_in, self.size_in)
        
    def forward(self, g, h):
        alpha = F.softmax(g.t().matmul(self.W(h.t()).t()), dim=1)
        return alpha.matmul(h.t())


