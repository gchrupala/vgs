import numpy
import logging
import python_speech_features as psf
import wav

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    mfcc_p = commands.add_parser('mfcc')
    mfcc_p.set_defaults(func=mfcc)
    mfcc_p.add_argument('--dataset', help='Dataset to process: flickr8k or places')

    args = parser.parse_args()
    args.func(args)    



def mfcc(args):
    files = get_wavs(args.dataset)
    output = []
    for f in file:
        logging.info("Extracting features from {}".format(f))
        output.append(extract_mfcc(f))
    numpy.save("{}.npy".format(args.dataset), output)
    

def extract_mfcc(f):
    (rate,sig) = wav.read(f)
    mfcc_feat = psf.mfcc(sig,rate)
    return numpy.asarray(mfcc_feat, dtype='float32')


def get_wavs(dataset):
    if dataset == "places":
        root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"
        names = [ root + "/" + line.split[1] for line in open(root + "/" + "metadata/utt2wav") ]
        return names[:20]
    elif dataset == "flickr8k":
        raise NotImplemented
    else
        raise ValueError ("Unknown dataset {}".format(dataset))

    
