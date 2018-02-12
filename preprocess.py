import numpy as np
import logging
import python_speech_features as psf
import scipy.io.wavfile as wav
import argparse
import onion.util as util
from vg.util import parse_map
import h5py

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    mfcc_p = commands.add_parser('mfcc')
    mfcc_p.set_defaults(func=mfcc)
    mfcc_p.add_argument('--dataset',  help='Dataset to process: flickr8k or places')
    mfcc_p.add_argument('--output',   help='Path to file where output will be saved', default='mfcc.h5')

    merge_p = commands.add_parser('merge')
    merge_p.set_defaults(func=merge)
    merge_p.add_argument('--prefix', help='Prefix of the input files')
    merge_p.add_argument('--output', help='Path to file where output will be saved', default='data.h5')
    args = parser.parse_args()
    args.func(args)    


def merge(args):
    root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"
    data = ( arr for i in range(230) for arr in np.load("{}.{}.npy".format(args.prefix, i)) )
    M = parse_map(open(root + "/" + "metadata/utt2wav"))
    with h5py.File(args.output, 'w') as f:
        for key, arr in zip(M.keys(), data):
            f.create_dataset(key, data=arr)
    

   
def mfcc_h5(args):
    if args.dataset != 'places':
        raise NotImplementedError
    with h5py.File(args.output, 'w') as f:
        root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"
        M = parse_map(open(root + "/" + "metadata/utt2wav"))        
        for key, wav in M.items():
            path =  root + "/" + wav
            arr = extract_mfcc(path)
            if arr is not None:
                f.create_dataset(key, data=arr)
   

def mfcc(args):
    if args.dataset == 'places':
        root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"    
        M = parse_map(open(root + "/metadata/utt2wav"))        
    elif args.dataset == 'flickr8k':
        def parse(lines):
            M = {}
            for line in lines:
                wav, jpg, no = line.split()
                M[wav] = "wavs/" + wav
            return M
        root = "/exp/gchrupal/corpora/flickr_audio/"
        M = parse(open(root + "/wav2capt.txt"))
    else:
        raise NotImplementedError
    D = {}
    for key, wav in M.items():
            path =  root + "/" + wav
            arr = extract_mfcc(path)
            if arr is not None:
                D[key] =arr
    np.save(args.output, D)
   
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
    
