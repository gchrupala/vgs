# Pytorch version of imaginet.audiovis_rhn
import numpy
import numpy as np
import torch.nn as nn
import torch
import onion.util as util
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
import onion.loss as loss
from onion import rhn, attention, conv
from vg.simple_data import vector_padder
from vg.scorer import Scorer, testing
from vg.defn.encoders import TextEncoderTop, TextEncoderBottom, SpeechEncoderBottom, SpeechEncoderTop, ImageEncoder
from vg.defn.decoders import DecoderWithAttn
from collections import Counter
import sys
import itertools
import json 

def step(task, *args):

    loss = task.train_cost(*args)
    task.optimizer.zero_grad()
    loss.backward()
    _ = nn.utils.clip_grad_norm(task.parameters(), task.config['max_norm'])
    return loss

class SpeechText(nn.Module):

    def __init__(self, speech_encoder, text_encoder, config):
        super(SpeechText, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.TextEncoderBottom = text_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        self.TextEncoderTop = TextEncoderTop(**config['TextEncoderTop'])  
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, speech, text):
        speech_enc = self.SpeechEncoderTop(self.SpeechEncoderBottom(speech))
        text_enc = self.TextEncoderTop(self.TextEncoderBottom(text))
        scores = loss.cosine_matrix(speech_enc, text_enc) 
        cost =  loss.contrastive(scores, margin=self.config['margin_size'])
        return cost            

    def args(self, item):
        return (item['audio'], item['input'].astype('int64'))

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)


class SpeechImage(nn.Module):

    def __init__(self, speech_encoder, config):
        super(SpeechImage, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        self.ImageEncoder = ImageEncoder(**config['ImageEncoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, speech, image):
        speech_enc = self.SpeechEncoderTop(self.SpeechEncoderBottom(speech))
        image_enc = self.ImageEncoder(image)
        scores = loss.cosine_matrix(speech_enc, image_enc) 
        cost =  loss.contrastive(scores, margin=self.config['margin_size'])
        return cost     

    def args(self, item):
        return (item['audio'], item['target_v'])

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)

class TextImage(nn.Module):

    def __init__(self, text_encoder, config):
        super(TextImage, self).__init__()
        self.config = config
        self.TextEncoderBottom = text_encoder
        self.TextEncoderTop = TextEncoderTop(**config['TextEncoderTop'])
        self.ImageEncoder = ImageEncoder(**config['ImageEncoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, text, image):
        text_enc = self.TextEncoderTop(self.TextEncoderBottom(text))
        image_enc = self.ImageEncoder(image)
        scores = loss.cosine_matrix(text_enc, image_enc) 
        cost =  loss.contrastive(scores, margin=self.config['margin_size'])
        return cost     

    def args(self, item):
        return (item['input'].astype('int64'), item['target_v'])

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)

    def encode_images(self, images):
        with testing(self):
            rep = self.ImageEncoder(images)
        return rep

    def predict(self, text):
        with testing(self):
            rep = self.TextEncoderTop(self.TextEncoderBottom(text))
        return rep

class SpeechTranscriber(nn.Module):

    def __init__(self, speech_encoder, config):
        super(SpeechTranscriber, self).__init__()
        self.config = config
        self.SpeechEncoderBottom = speech_encoder
        self.SpeechEncoderTop = SpeechEncoderTop(**config['SpeechEncoderTop'])
        self.TextDecoder = DecoderWithAttn(**config['TextDecoder'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def cost(self, speech, target, target_prev):
        states, rep = self.SpeechEncoderTop.states(self.SpeechEncoderBottom(speech))
        target_logits = self.TextDecoder(states, rep, target_prev)
        cost =  F.cross_entropy(target_logits.view(target_logits.size(0)*target_logits.size(1),-1), 
                                target.view(target.size(0)*target.size(1)))
        return cost            

    def args(self, item):
        return (item['audio'], item['target_t'].astype('int64'), item['target_prev_t'].astype('int64'))

    def test_cost(self, *args):
        with testing(self):
            return self.cost(*args)

class Net(nn.Module):

    def __init__(self, config):
        super(Net, self).__init__()
        self.SpeechEncoderBottom = SpeechEncoderBottom(**config['SpeechEncoderBottom'])
        self.TextEncoderBottom = TextEncoderBottom(**config['TextEncoderBottom']) \
                                if config.get('TextEncoderBottom') else None
        self.SpeechText  = SpeechText(self.SpeechEncoderBottom, self.TextEncoderBottom, config['SpeechText'])  \
                                if config.get('SpeechText') else None
        self.SpeechImage = SpeechImage(self.SpeechEncoderBottom, config['SpeechImage'])
        self.TextImage   = TextImage(self.TextEncoderBottom, config['TextImage']) \
                                if config.get('TextImage')  else None
        self.SpeechTranscriber = SpeechTranscriber(self.SpeechEncoderBottom, config['SpeechTranscriber']) \
                                if config.get('SpeechTranscriber') else None
  
    def encode_images(self, images):
        with testing(self):
            rep = self.SpeechImage.ImageEncoder(images)
        return rep

    def predict(self, audio):
        with testing(self):
            rep = self.SpeechImage.SpeechEncoderTop(self.SpeechImage.SpeechEncoderBottom(audio))
        return rep


def encode_texts(task, texts, batch_size=128):
        return numpy.vstack([ task.TextImage.predict(
                            torch.autograd.Variable(torch.from_numpy(
                                task.batcher.batch_inp(task.mapper.transform(batch)).astype('int64'))).cuda()).data.cpu().numpy()
                            for batch in util.grouper(texts, batch_size) ])

def encode_images_TextImage(task, imgs, batch_size=128):
    """Project imgs to the joint space using model.
    """
    return numpy.vstack([ task.TextImage.encode_images(
                            torch.autograd.Variable(torch.from_numpy(
                                numpy.vstack(batch))).cuda()).data.cpu().numpy()
                          for batch in util.grouper(imgs, batch_size) ])

def experiment(net, data, run_config):
    def valid_loss(name, task):
        result = []
        for item in data[name].iter_valid_batches():
            args = task.args(item)
            args = [torch.autograd.Variable(torch.from_numpy(x), volatile=True).cuda() for x in args ]
            result.append([ x.data.cpu().numpy() for x in task.test_cost(*args) ])
        return result
    
    net.cuda()
    net.train()
    scorer = run_config['Scorer']
    last_epoch = 0

    for _, task in run_config['tasks']:
        task.optimizer.zero_grad()

    with open("result.json", "w") as out:
      for epoch in range(last_epoch+1, run_config['epochs'] + 1):
        costs = dict(SpeechText=Counter(), SpeechImage=Counter(), TextImage=Counter(), SpeechTranscriber=Counter())
        
        for _j, items in enumerate(zip(data['SpeechImage'].iter_train_batches(reshuffle=True), 
                                       data['SpeechText'].iter_train_batches(reshuffle=True),
                                       data['TextImage'].iter_train_batches(reshuffle=True),
                                       data['SpeechTranscriber'].iter_train_batches(reshuffle=True))):
            j = _j + 1  
            item = dict(SpeechImage=items[0], SpeechText=items[1], TextImage=items[2], SpeechTranscriber=items[3])
            for name, task in run_config['tasks']:
                spk = item[name]['speaker'][0] if len(set(item[name]['speaker'])) == 1 else 'MIXED'
                args = task.args(item[name])
                args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
              

                loss = task.cost(*args)
                task.optimizer.zero_grad()
                loss.backward()
                _ = nn.utils.clip_grad_norm(task.parameters(), task.config['max_norm'])
                task.optimizer.step()
                costs[name] += Counter({'cost':loss.data[0], 'N':1})
                print(epoch, j, j*data[name].batch_size, name, spk, "train", "".join([str(costs[name]['cost']/costs[name]['N'])]))

                if j % run_config['validate_period'] == 0:
                    loss = valid_loss(name, task)
                    print(epoch, j, 0, name, "VALID", "valid", "".join([str(numpy.mean(loss))]))


                sys.stdout.flush()
        torch.save(net, "model.{}.pkl".format(epoch))

        with testing(net):
            result = dict(epoch=epoch, 
                          rsa=scorer.rsa_image(net), 
                          retrieval=scorer.retrieval(net), 
                          speaker_id=scorer.speaker_id(net))
            out.write(json.dumps(result))
            out.write("\n")
            out.flush()


    torch.save(net, "model.pkl")

