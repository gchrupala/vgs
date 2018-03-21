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
from vg.defn.simple_encoder import Encoder
from vg.scorer import Scorer, testing
from collections import Counter
import sys
import json

class Decoder(nn.Module):
    def __init__(self, size_feature, size, depth=1):
        super(Decoder, self).__init__()
        util.autoassign(locals())
        self.h0 = torch.autograd.Variable(torch.zeros(self.depth, 1, self.size))
        self.RNN = nn.GRU(self.size_feature, self.size, self.depth, batch_first=True)
        self.Proj = nn.Linear(self.size, self.size_feature)
        
    def forward(self, rep, prev):
        rep = rep.expand(self.depth, rep.size(0), rep.size(1))
        out, last = self.RNN(prev, rep)
        pred = self.Proj(out)
        return pred

def step(task, *args):
    loss = task.train_cost(*args)
    task.optimizer.zero_grad()
    loss.backward()
    _ = nn.utils.clip_grad_norm(task.parameters(), task.config['audio']['max_norm'])
    return loss


class Audio(nn.Module):

    def __init__(self, config):
        super(Audio, self).__init__()
        util.autoassign(locals())
        self.Encode = Encoder(**config['encoder'])

        self.Decode1 = Decoder(config['audio']['size_feature'], config['audio']['size'])

        self.Decode3 = Decoder(config['audio']['size_feature'], config['audio']['size'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['audio']['lr'])
    def step(self, *args):
        return step(self, *args)

    def forward(self, speech2):
        rep = F.normalize(self.Encode(speech2), p=2, dim=1)
        return rep
    
    def cost(self, speech1_prev, speech1, rep, speech3_prev, speech3):
        pred1 = self.Decode1(rep, speech1_prev)
        pred3 = self.Decode3(rep, speech3_prev)
        return F.mse_loss(pred1, speech1) + F.mse_loss(pred3, speech3)

    def train_cost(self, speech1_prev, speech1, speech2, speech3_prev, speech3):
        rep = self(speech2)
        return self.cost(speech1_prev, speech1, rep, speech3_prev, speech3)

    def test_cost(self, speech1_prev, speech1, speech2, speech3_prev, speech3):
        with testing(self):
            self.eval()
            rep = self(speech2)
            return self.cost(speech1_prev, speech1, rep, speech3_prev, speech3)

    def predict(self, speech):
        with testing(self):
            return self(speech)


    def args(self, item):
        return (item['audio_1_prev'], item['audio_1'], item['audio_2'], item['audio_3_prev'], item['audio_3'])


def experiment(net, data, prov, model_config, run_config):
    def valid_loss(task):
        result = []
        for item in data.iter_valid_batches():
            args = task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append(task.test_cost(*args).data.cpu().numpy())
        return result
    
    net.cuda()
    net.train()
    scorer = Scorer(prov, dict(split='val', tokenize=lambda x: x['audio'], batch_size=data.batch_size))
    net.optimizer.zero_grad()
    last_epoch = 0
    with open("result.json", "w") as out:
      for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        costs = Counter()
        net.train()
        for _j, item in enumerate(data.iter_train_batches()):
                j = _j + 1
                name = "Aud"; task = net
                spk = item['speaker'][0] if len(set(item['speaker'])) == 1 else 'MIXED'
                args = task.args(item)
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
                loss = task.optimizer.step(lambda: task.step(*args))
                costs += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data.batch_size, name, spk, "train", "".join([str(costs['cost']/costs['N'])]))
                if j % run_config['validate_period'] == 0:
                    print(epoch, j, 0, name, spk, "valid", "".join([str(numpy.mean(valid_loss(task)))]))
                sys.stdout.flush()
        torch.save(net, "model.{}.pkl".format(epoch))
        with testing(net):
            result = dict(epoch=epoch, rsa=scorer.rsa_image(net), 
                                       para=scorer.retrieval_para(net))
            if run_config.get('speaker_id', True):
                 result['speaker_id']=scorer.speaker_id(net)
            out.write(json.dumps(result))
            out.write("\n")
            out.flush()
    torch.save(net, "model.pkl")


