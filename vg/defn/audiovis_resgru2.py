# Pytorch version of imaginet.audiovis_rhn
import numpy
import numpy as np
import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from onion import rhn, attention, conv, stacked_gru
from vg.simple_data import vector_padder



class Encoder(nn.Module):

    def __init__(self, size_vocab, size, depth=1, recur_depth=1, bidirectional=False,
                 filter_length=6, filter_size=64, stride=2, drop_i=0.75 , drop_s=0.25):
        super(Encoder, self).__init__()
        util.autoassign(locals())
        #self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))

        self.Conv = conv.Convolution1D(self.size_vocab, self.filter_length, self.filter_size, stride=self.stride)
        # self.RNN = nn.GRU(self.filter_size, self.size, self.depth, batch_first=True)
        self.RNN = stacked_gru.StackedGRU(self.filter_size, self.size, self.depth,
                     bidirectional=bidirectional, residual=True, batch_first=True)

    def forward(self, input):
        return self.RNN(self.Conv(input))#, self.h0.expand(self.depth, input.size(0), self.size).cuda())


class Visual(nn.Module):

    def __init__(self, config):
        super(Visual, self).__init__()
        util.autoassign(locals())
        self.margin_size = config.get('margin_size', 0.2)
        # FIXME FIXME ADD gradient clipping!
        #self.make_updater = lambda: optim.Adam(self.parameters(), lr=config['lr'])
        self.max_norm = config['max_norm']
        self.Encode = Encoder(config['size_vocab'],
                              config['size'],
                              filter_length=config.get('filter_length', 6),
                              filter_size=config.get('filter_size', 1024),
                              stride=config.get('stride', 3),
                              depth=config.get('depth', 1),
                              bidirectional=config.get('bidirectional', False))
        self.Attn   = attention.SelfAttention(config['size'], size=config.get('size_attn', 512))
        self.ImgEncoder  = util.make_linear(config['size_target'], config['size'])


    def forward(self, speech):
        return F.normalize(self.Attn(self.Encode(speech)), p=2, dim=1)

    def cost(self, speech, image):
        s_encoded = F.normalize(self.Attn(self.Encode(speech)), p=2, dim=1)
        i_encoded = F.normalize(self.ImgEncoder(image), p=2, dim=1)
        scores = cosine_matrix(s_encoded, i_encoded) 
        return contrastive2(scores, margin=self.margin_size)

    def train_cost(self, speech, image):
        return self.cost(speech, image)

    def test_cost(self, speech, image):
        mode = self.training
        self.eval()
        cost = self.cost(speech, image)
        self.training = mode
        return cost

    def predict(self, speech):
        mode = self.training
        self.eval()
        pred = self(speech)
        self.training = mode
        return pred

    def args(self, item):
        return (item['audio'], item['target_v'])

    def encode_images(self, images):
        mode = self.training
        self.eval()
        rep = F.normalize(self.ImgEncoder(images), p=2, dim=1)
        self.training = mode
        return rep

def contrastive2(score_matrix, margin=0.2):
        # i: (fixed) image embedding,
        # s: sentence embedding
        errors = - score_matrix
        diagonal = diag(errors)
        # compare every diagonal score to scores in its column (all contrastive images for each sentence)
        cost_s = torch.clamp(margin - errors + diagonal, min=0)
        # all contrastive sentences for each image
        cost_i = torch.clamp(margin - errors + diagonal.view(-1, 1), min=0)
        cost_tot = cost_s + cost_i
        # clear diagonals
        I = torch.autograd.Variable(torch.eye(cost_tot.size(0)), requires_grad=True).cuda()
        cost_tot = (1-I) * cost_tot
        return cost_tot.mean()

def contrastive(i, s, margin=0.2):
        # i: (fixed) image embedding,
        # s: sentence embedding
        errors = - cosine_matrix(i, s)
        diagonal = diag(errors)
        # compare every diagonal score to scores in its column (all contrastive images for each sentence)
        cost_s = torch.clamp(margin - errors + diagonal, min=0)
        # all contrastive sentences for each image
        cost_i = torch.clamp(margin - errors + diagonal.view(-1, 1), min=0)
        cost_tot = cost_s + cost_i
        # clear diagonals
        I = torch.autograd.Variable(torch.eye(cost_tot.size(0)), requires_grad=True).cuda()
        cost_tot = (1-I) * cost_tot
        return cost_tot.mean()

def diag(M):
    """Return the diagonal of the matrix."""
    I = torch.autograd.Variable(torch.eye(M.size(0)), requires_grad=True).cuda()
    return (M * I).sum(dim=0)


def cosine_matrix(U, V):
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())

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

def encode_images(model, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ model.task.encode_images(torch.autograd.Variable(torch.from_numpy(numpy.vstack(batch))).cuda()).data.cpu().numpy()
                          for batch in util.grouper(imgs, batch_size) ])

def symbols(model):
    return model.batcher.mapper.ids.decoder
