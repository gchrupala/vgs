import gentle
import json
import multiprocessing
import logging


def getwav2text():
    id2text = {}
    for line in open("../data/flickr8k/Flickr8k.token.txt"):
        fields = line.split()
        id2text[fields[0]] = ' '.join(fields[1:])
    wav2id = {}
    for line in open("../data/flickr8k/wav2capt.txt"):
        fields = line.split()
        wav2id[fields[0]] = ''.join(fields[1:])
    wav2text = {}
    for w, ID in wav2id.items():
        wav2text[w] = id2text[ID]
    return wav2text
    
def align_flickr8k():
    logging.getLogger().setLevel("INFO")
    nthreads = multiprocessing.cpu_count()
    wavroot = "/home/gchrupala/repos/vgs/data/flickr8k/"
    DATA = getwav2text()
    with open("../data/flickr8k/dataset.fa.json", 'w') as fa:
        for wavp, text in DATA.items():
            result = align("../data/flickr8k/flickr_audio/wavs/{}".format(wavp), text, nthreads=nthreads)
            fa.write(json.dumps(json.loads(result.to_json())))
            fa.write("\n")


def on_progress(p):
    for k,v in p.items():
        logging.debug("%s: %s" % (k, v))


def align(audiopath, text, nthreads=1):
    resources = gentle.Resources()
    logging.info("converting audio to 8K sampled wav")
    with gentle.resampled(audiopath) as wavfile:
        logging.info("Starting alignment")
        aligner = gentle.ForcedAligner(resources, text, nthreads=nthreads, disfluency=False, 
                                        conservative=False)
        return aligner.transcribe(wavfile, progress_cb=on_progress, logging=logging)    


