import numpy
import random
seed = 1235
random.seed(seed)
numpy.random.seed(seed)

import vg.simple_data as sd
import vg.experiment as E
import vg.flickr8k_provider as dp_f
import vg.libri_provider as dp_l

import vg.defn.audiotext_gru_partial as D

batch_size = 14
epochs=25

prov_flickr = dp_f.getDataProvider('flickr8k', root='../..', audio_kind='mfcc')
prov_libri = dp_l.getDataProvider('libri', root='../..', audio_kind='mfcc')

data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1, scale=False,
                            batch_size=batch_size, shuffle=True)
data_libri  = sd.SimpleData(prov_libri, tokenize=sd.characters, min_df=1, scale=False,
                            batch_size=batch_size, shuffle=True)

model_config = dict(Image=dict(encoder=dict(size=1024, size_target=4096),
                               lr=0.0002,
                               margin_size=0.2,
                               max_norm=2.0, 
                               SpeechEncoderTop=dict(size=1024, size_input=1024, depth=2, size_attn=128)),
                    Text=dict(encoder=dict(size_feature=data_libri.mapper.size(),
                                           size_embed=128,
                                           size=1024,
                                           depth=2,
                                           size_attn=128),
                              SpeechEncoderTop=dict(size=1024,
                                                    size_input=1024,
                                                    depth=0,
                                                    size_attn=128), 
                              lr=0.0002,
                              margin_size=0.2,
                              max_norm=2.0),
                    SpeechEncoderBottom=dict(size=1024, depth=2, size_vocab=13, filter_length=6, filter_size=64, stride=2)
                   )






def audio(sent):
    return sent['audio']

net = D.Net(model_config)
run_config = dict(epochs=epochs, validate_period=400, tasks=[('Text', net.Text), ('Image', net.Image)])
D.experiment(net=net, data=dict(Text=data_libri, Image=data_flickr), run_config=run_config)
