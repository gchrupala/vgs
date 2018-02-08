import numpy as np
import logging
import python_speech_features as psf
import scipy.io.wavfile as wav
import argparse
import onion.util as util

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    mfcc_p = commands.add_parser('mfcc')
    mfcc_p.set_defaults(func=mfcc)
    mfcc_p.add_argument('--dataset',  help='Dataset to process: flickr8k or places')
    mfcc_p.add_argument('--output',   help='Path to file where output will be saved', default='mfcc.npy')

    merge_p = commands.add_parser('merge')
    merge_p.set_defaults(func=merge)
    merge_p.add_argument('--prefix', help='Prefix of the input files')
    args = parser.parse_args()
    args.func(args)    

def load_places(prefix):
    data = [ arr for i in range(203) for arr in np.load("{}.{}.npy".format(prefix, i)) ]
    wavs = get_wavs('places')
    return dict(zip(wavs, data))
   
def mfcc(args):
    files = get_wavs(args.dataset)
    output = map(extract_mfcc, files)    
    for i, chunk in enumerate(util.grouper(output, 1000)):
        np.save("{}.{}".format(args.output, i), chunk)
   
import time
import timeout_decorator
@timeout_decorator.timeout(5)
def extract_mfcc(f, truncate=20):
    logging.info("Extracting features from {}".format(f))
    try:
        (rate, sig) = wav.read(f)
        max_len = truncate*rate
    except:
        logging.warning("Error reading file {}".format(f))
        return None      
    try:
        mfcc_feat = psf.mfcc(sig[:max_len], rate)    
        return np.asarray(mfcc_feat, dtype='float32')

    except:
        logging.warning("Error extracting features from file {}".format(f))
        return None

def get_wavs(dataset):
    if dataset == "places":
        root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"
        names = [ root + "/" + line.split()[1] for line in open(root + "/" + "metadata/utt2wav") ]
        return names
    elif dataset == "flickr8k":
        raise NotImplementedError
    else:
        raise ValueError ("Unknown dataset {}".format(dataset))



if __name__ == '__main__':
    main()
    
