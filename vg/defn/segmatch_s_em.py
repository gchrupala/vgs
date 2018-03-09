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
from vg.defn.encoder_em import Encoder
from vg.scorer import Scorer, testing
from collections import Counter
import sys
from onion.loss import cosine_matrix, contrastive 

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
        self.ProjBeg = nn.Linear(config['audio']['size'], config['audio']['size_target'])
        self.ProjEnd = nn.Linear(config['audio']['size'], config['audio']['size_target'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['audio']['lr'])

    def step(self, *args):
        return step(self, *args)

    def score(self, x, y):
        return F.cosine_similarity(x, y, dim=1)
        
    def forward(self, speech):
        return F.normalize(self.ProjBeg(self.Encode(speech)), p=2, dim=1)

    def cost(self, beg, end):
        beg_encoded = F.normalize(self.ProjBeg(self.Encode(beg)), p=2, dim=1)
        end_encoded = F.normalize(self.ProjEnd(self.Encode(end)), p=2, dim=1)
        scores = cosine_matrix(beg_encoded, end_encoded) 
        return contrastive(scores, margin=self.config['audio']['margin_size'])

    def train_cost(self, beg, end):
        return self.cost(beg, end)

    def test_cost(self, beg, end):
        with testing(self):
            self.eval()
            cost = self.cost(beg, end)
        return cost

    def predict(self, speech):
        with testing(self):
            self.eval()
            pred = self(speech)
        return pred

    def args(self, item):
        beg = item['input_beg'].astype('int64')
        end = item['input_end'].astype('int64')
        assert min(beg.shape) > 0 and min(end.shape) > 0, "Broken input {}".format(item['input'])
        return (beg, end)

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

    net.optimizer.zero_grad()
    last_epoch = 0
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
    torch.save(net, "model.pkl")



