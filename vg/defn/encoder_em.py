import torch
import torch.nn as nn
from onion import attention, conv
import onion.util as util

class Encoder(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, size_attn=512, dropout_p=0.0):
        super(Encoder, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) 
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return self.Attn(out)

