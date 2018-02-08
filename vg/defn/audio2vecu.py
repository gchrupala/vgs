# Pytorch version of imaginet.audiovis_rhn
import numpy
import numpy as np
import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from onion import rhn, attention, conv
from vg.simple_data import vector_padder



class Encoder(nn.Module):

    def __init__(self, size_vocab, size, depth=1, 
                 filter_length=6, filter_size=64, stride=2, residual=False):
        super(Encoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))

        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride, padding=0)

        self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)

    def forward(self, input):

        out, last = self.RNN(self.Conv(input), self.h0.expand(self.depth, input.size(0), self.size).cuda())
        return out

class Decoder(nn.Module):
    def __init__(self, size_vocab, size, depth=1):
        super(Decoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.RNN = nn.GRU(self.size, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_vocab)
        
    def forward(self, rep, target):

        R = rep.unsqueeze(1).expand(-1, target.size(1), -1).cuda()
        
        H0 = self.h0.expand(self.depth, target.size(0), self.size).cuda()        
        out, last = self.RNN(R, H0)
        pred = self.Proj(out)
        return pred

class Audio(nn.Module):

    def __init__(self, config):
        super(Audio, self).__init__()
        util.autoassign(locals())
        # FIXME FIXME ADD gradient clipping!
        #self.make_updater = lambda: optim.Adam(self.parameters(), lr=config['lr'])
        self.max_norm = config['max_norm']
        self.Encode = Encoder(config['size_vocab'],
                              config['size'],
                              filter_length=config.get('filter_length', 6),
                              filter_size=config.get('filter_size', 1024),
                              stride=config.get('stride', 3),
                              depth=config.get('depth', 1),
                              residual=config.get('residual', False))
        self.Attn   = attention.SelfAttention(config['size'], size=config.get('size_attn', 512))

        self.Decode1 = Decoder(config['size_vocab'], 
                               config['size'])

        self.Decode3 = Decoder(config['size_vocab'], 
                               config['size'])


    def forward(self, speech2):
        rep = F.normalize(self.Attn(self.Encode(speech2)), p=2, dim=1)
        return rep
    
    def cost(self, speech1_prev, speech1, rep, speech3_prev, speech3):
        pred1 = self.Decode1(rep, speech1)
        pred3 = self.Decode3(rep, speech3)
        return F.mse_loss(pred1, speech1) + F.mse_loss(pred3, speech3)

    def train_cost(self, speech1_prev, speech1, speech2, speech3_prev, speech3):
        rep = self(speech2)
        return self.cost(speech1_prev, speech1, rep, speech3_prev, speech3)

    def test_cost(self, speech1_prev, speech1, speech2, speech3_prev, speech3):
        mode = self.training
        self.eval()
        rep = self(speech2)
        cost = self.cost(speech1_prev, speech1, rep, speech3_prev, speech3)
        self.training = mode
        return cost

    def predict(self, speech):
        mode = self.training
        self.eval()
        pred = self(speech)
        self.training = mode
        return pred

    def args(self, item):
        return (item['audio_1_prev'], item['audio_1'], item['audio_2'], item['audio_3_prev'], item['audio_3'])



def encode_sentences(model, audios, batch_size=128):
    """Project audios to the joint space using model.

    For each audio returns a vector.
    """
    return numpy.vstack([ model.task.predict(torch.autograd.Variable(torch.from_numpy(vector_padder(batch))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(audios, batch_size) ])

def iter_layer_states(model, audios, batch_size=128):
    """Pass audios through the model and for each audio return the state of each timestep and each layer."""

    lens = (numpy.array(map(len, audios)) + model.config['filter_length']) // model.config['stride']
    rs = (r for batch in util.grouper(audios, batch_size) for r in model.task.pile(vector_padder(batch)))
    for (r,l) in itertools.izip(rs, lens):
         yield r[-l:,:,:]

def layer_states(model, audios, batch_size=128):
    return list(iter_layer_states(model, audios, batch_size=128))


def symbols(model):
    return model.batcher.mapper.ids.decoder
