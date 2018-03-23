import torch
import torch.nn as nn
from onion import attention, conv
import onion.util as util

class Encoder(nn.Module):

    def __init__(self, size_feature, size, depth=1, filter_length=6, filter_size=64, stride=2, size_attn=512, dropout_p=0.0):
        super(Encoder, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.Conv = conv.Convolution1D(self.size_feature, self.filter_length, self.filter_size, stride=self.stride, padding=0)
        self.RNN  = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, speech):
        h0 = self.h0.expand(self.depth, speech.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Conv(speech)), h0)
        return self.Attn(out)

