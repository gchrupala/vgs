import torch
import torch.nn as nn
from onion import attention, conv
import onion.util as util
import torch.nn.functional as F

def l2normalize(x):
    return F.normalize(x, p=2, dim=1)

#LEGACY
class TextEncoder(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, size_attn=512, dropout_p=0.0):
        super(TextEncoder, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) 
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return l2normalize(self.Attn(out))


class TextEncoderBottom(nn.Module):

    def __init__(self, size_feature, size, size_embed=64, depth=1, dropout_p=0.0):
        super(TextEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) 
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN  = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)

    
    def forward(self, text):
        h0 = self.h0.expand(self.depth, text.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Embed(text)), h0)
        return out


class TextEncoderTop(nn.Module):

    def __init__(self, size_feature, size, depth=1, size_attn=512, dropout_p=0.0):
        super(TextEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0   = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN  = nn.GRU(self.size_feature, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
    
    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))


class SpeechEncoder(nn.Module):

    def __init__(self, size_vocab, size, depth=1, filter_length=6, filter_size=64, stride=2, size_attn=512, dropout_p=0.0):
        super(SpeechEncoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        self.Dropout = nn.Dropout(p=self.dropout_p)
        self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)

    def forward(self, input):
        h0 = self.h0.expand(self.depth, input.size(0), self.size).cuda()
        out, last = self.RNN(self.Dropout(self.Conv(input)), h0)
        return l2normalize(self.Attn(out))


class SpeechEncoderBottom(nn.Module):
    def __init__(self, size_vocab, size, depth=1, filter_length=6, filter_size=64, stride=2, dropout_p=0.0):
        super(SpeechEncoderBottom, self).__init__()
        util.autoassign(locals())
        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, last = self.RNN(self.Dropout(self.Conv(x)), h0)
        else:
            out = self.Conv(x)
        return out

class SpeechEncoderTop(nn.Module):
    def __init__(self, size_input, size, depth=1, size_attn=512, dropout_p=0.0):
        super(SpeechEncoderTop, self).__init__()
        util.autoassign(locals())
        if self.depth > 0:
            self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
            self.Dropout = nn.Dropout(p=self.dropout_p)
            self.RNN = nn.GRU(self.size_input, self.size, self.depth, batch_first=True)
        self.Attn = attention.SelfAttention(self.size, size=self.size_attn)
        

    def forward(self, x):
        if self.depth > 0:
            h0 = self.h0.expand(self.depth, x.size(0), self.size).cuda()
            out, _last = self.RNN(self.Dropout(x), h0)
        else:
            out = x
        return l2normalize(self.Attn(out))

    

class ImageEncoder(nn.Module):
    
    def __init__(self, size, size_target):
        super(ImageEncoder, self).__init__()
        self.Encoder = util.make_linear(size_target, size)
    
    def forward(self, img):
        return l2normalize(self.Encoder(img))


