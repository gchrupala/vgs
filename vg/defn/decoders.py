import torch
import torch.nn as nn
import onion.util as util
import torch.nn.functional as F

# BAHDANAU-STYLE ATTENTION

class BilinearAttention(nn.Module):
    """Soft alignment between two sequences, aka attention. Based on https://arxiv.org/abs/1508.04025v5."""

    def __init__(self, size_in): 
        super(BilinearAttention, self).__init__()
        util.autoassign(locals())
        self.W = nn.Linear(self.size_in, self.size_in)
        
    def forward(self, g, h):
        alpha = F.softmax(g.bmm(self.W(h).permute(0,2,1)), dim=1) # FIXME is dim=1 correct?
        # Alignment based on bi-linear scores between g (source) and h (target)
        context = alpha.permute(0,2,1).bmm(g)
        # print(context.size(), h.size())
        return context

class SimpleDecoder(nn.Module):
    """Simple decoder."""
    def __init__(self, size_feature, size, size_embed=64, depth=1):
        super(SimpleDecoder, self).__init__()
        util.autoassign(locals())
        self.Embed  = nn.Embedding(self.size_feature, self.size_embed) # Why not share embeddings with encoder?
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.RNN = nn.GRU(self.size_embed, self.size, self.depth, batch_first=True)


    def forward(self, prev, rep): 
        R = rep.expand(self.depth, rep.size(0), rep.size(1))
        out, last = self.RNN(self.Embed(prev), R)
        return out, last

# Let's look at this post
# https://stackoverflow.com/questions/50571991/implementing-luong-attention-in-pytorch

# Original Matlab code by Luong is here: https://github.com/lmthang/nmt.hybrid

class DecoderWithAttn(nn.Module):

    def __init__(self, size, size_target_vocab, depth=1):
        super(DecoderWithAttn, self).__init__()
        util.autoassign(locals())
        self.Decoder = SimpleDecoder(self.size_target_vocab, self.size, depth=self.depth)
        self.BAttn = BilinearAttention(self.size)
        self.Proj = nn.Linear(self.size * 2, self.size_target_vocab)
    
    def forward(self, states, rep, prev):
        # Encoder returns the hidden state per time step, states, and a global representation, rep
        # Decoder decodes conditioned on rep, and on the symbol decoded at previous time step, prev
        h, _last = self.Decoder(prev, rep)
        # Bilinear attention generates a weighted sum of the source hidden states (context) for each time 
        # step of the target
        context = self.BAttn(states, h)
        # The context is concatenated with target hidden states
        h_context = torch.cat((context, h), dim=2)
        # The target symbols are generated conditioned on the concatenation of target states and context
        pred = self.Proj(h_context)
        #print(pred.size())
        return pred


        
# Conditioned on source and on target at t-1
class CondDecoder(nn.Module):
    def __init__(self, size_feature, size, depth=1):
        super(CondDecoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.RNN = nn.GRU(self.size_feature, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_feature)
        
    def forward(self, rep, prev):
        rep = rep.expand(self.depth, rep.size(0), rep.size(1))
        out, last = self.RNN(prev, rep)
        pred = self.Proj(out)
        return pred

# Conditioned only on source
class UncondDecoder(nn.Module):
    def __init__(self, size_feature, size, depth=1):
        super(UncondDecoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.RNN = nn.GRU(self.size, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_feature)

    def forward(self, rep, target):
        R = rep.unsqueeze(1).expand(-1, target.size(1), -1).cuda()      
        H0 = self.h0.expand(self.depth, target.size(0), self.size).cuda()        
        out, last = self.RNN(R, H0)
        pred = self.Proj(out)
        return pred


