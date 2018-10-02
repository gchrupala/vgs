def search(prefixes, getnext, stop, value, K=1):
    """Beam search: maintain a collection of K prefixes, sorted by value."""
    def extend(prefix):
        return [ prefix + [item] for item in getnext(prefix)]
    extended = [   item  for prefix in prefixes for item in extend(prefix) ]
    reranked = list(reversed(sorted(extended, key=value)))
    pruned =   reranked[:K]
    if all([stop(prefix) for prefix in pruned]):
        return pruned
    else:
        return search(pruned, getnext, stop, value, K)

# Some toy example    

def value(prefix):
    return sum(val for val, _ in prefix)


import torch


def transcribe(net, audio, K=1, BEG=0, END=1, maxlen=20):
    task = net.SpeechTranscriber    
    states, rep = task.SpeechEncoderTop.states(task.SpeechEncoderBottom(audio)) 
    def stop(prefix):                                                                     
        return prefix[-1] == END or len(prefix) > maxlen

    def getnext(prefix):
        ids = [ i for _, i in prefix ]
        last = ids[-1]
        print(ids)
        prev = torch.autograd.Variable(torch.LongTensor([[last]])).cuda()
        logits = task.TextDecoder(states, rep, prev).squeeze()
        logprobs = torch.nn.functional.log_softmax(logits, dim=0)
        return [ (val, i) for (i, val) in enumerate(logprobs.cpu().data.numpy()) if i != BEG ]
 

    prefixes = [[(0.0, BEG)]]
    return search(prefixes, getnext, stop, value, K=K)


def main():
    import vg.flickr8k_provider as dp_f 
    import vg.simple_data as sd
    batch_size = 16
    prov_flickr = dp_f.getDataProvider('flickr8k', root='.', audio_kind='mfcc')
    data_flickr = sd.SimpleData(prov_flickr, tokenize=sd.characters, min_df=1, scale=False,
                                     batch_size=batch_size, shuffle=False)
    net = torch.load("experiments/s2-t.-s2i2-s2t.-t2s.-s2d0-d1-embed-128-joint-e/model.23.pkl")
    #net = torch.load("experiments/s2-t1-s2i2-s2t0-t2s0-s2d0-d1-joint-f/model.19.pkl")
    net.SpeechTranscriber.TextDecoder.Decoder.RNN.flatten_parameters()
    net.SpeechTranscriber.SpeechEncoderBottom.RNN.flatten_parameters()
    batches = data_flickr.iter_valid_batches() 
    first = next(batches) 
    texts = list(data_flickr.mapper.inverse_transform(first['input']))
    args = net.SpeechTranscriber.args(first) 
    args = [torch.autograd.Variable(torch.from_numpy(x)).cuda() for x in args ]
    audio, target_t, target_prev_t = args
    for j in range(10):
        print(''.join(texts[j]))
        for seq in transcribe(net, audio[j:j+1], K=5, maxlen=20):
            
            vals, ids = zip(*seq)
            
            chars = list(data_flickr.mapper.inverse_transform([ids]))[0]
            text =  ''.join([ '_' if char == '<BEG>' else '|' if char == '<END>' else char for char in chars])
            print("{:.2f} {}".format(sum(vals), text))
        print()



    
main()

    
