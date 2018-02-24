import random
import numpy
seed =1235
random.seed(seed)
numpy.random.seed(seed)
    
import vg.simple_data as sd
import vg.experiment as E
import vg.vendrov_provider as dp
import vg.defn.segmatch as D
dataset = 'coco'
batch_size = 32
epochs=15
prov = dp.getDataProvider(dataset, root='../..', audio_kind='mfcc')
data = sd.SimpleData(prov, min_df=10, scale=False,
                     batch_size=batch_size, shuffle=True, erasure=15, limit=5000)
print("Loaded data")
model_config = dict(size=512, depth=5,  max_norm=2.0, residual=True,
                    lr=0.0002, size_vocab=13, size_target=512,
                    filter_length=6, filter_size=64, stride=3,
                    contrastive=True, margin_size=0.2, fixed=True, 
                    init_img='xavier', size_attn=512)
run_config = dict(seed=71, task=D.Audio, epochs=epochs, validate_period=4000)



def audio(sent):
    return sent['audio']

eval_config = dict(tokenize=audio, split='val', task=D.Audio, batch_size=batch_size,
                   epochs=epochs, encode_sentences=D.encode_sentences, para=True)

E.run_train(data, prov, model_config, run_config, eval_config, resume=False)
