import os
import csv
import numpy as np
import logging
import python_speech_features as psf
# import scipy.io.wavfile as wav
import soundfile as sf
import argparse
from vg.util import parse_map
import h5py
from extract_img_feats import img_features
import sys
from tqdm import tqdm

def main():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    mfcc_p = commands.add_parser('mfcc')
    mfcc_p.set_defaults(func=mfcc)
    mfcc_p.add_argument('--dataset',  help='Dataset to process: flickr8k, places or librispeech')
    mfcc_p.add_argument('--output',   help='Path to file where output will be saved', default='mfcc.h5')

    merge_p = commands.add_parser('merge')
    merge_p.set_defaults(func=merge)
    merge_p.add_argument('--prefix', help='Prefix of the input files')
    merge_p.add_argument('--output', help='Path to file where output will be saved', default='data.h5')
    
    imgfeats_p = commands.add_parser("imgfeats") 
    imgfeats_p.set_defaults(func=run_img_feats)
    imgfeats_p.add_argument('--dataset', choices=['flickr8k', 'places'],  help='Dataset to process', default='places')
    imgfeats_p.add_argument('--output',   help='Path to file where output will be saved', default='imgfeats')
    imgfeats_p.add_argument('--cnn', choices=["vgg16", "vgg19", "hybrid"],  help='Which CNN to run', default="vgg16")
    imgfeats_p.add_argument('--resize', type=int,  help='Integer, resize images shorter side to this size', default=None)
    imgfeats_p.add_argument('--crop_size', type=int,  help='Integer, crop this sized square from the images', default=224)
    imgfeats_p.add_argument('--tencrop', action='store_true',  help='If given, average over 10 crop features.')
    
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
    # Produces .npy file, but also a .csv with meta-data
    elif args.dataset == "librispeech":
        root = "/roaming/u1257964/LibriSpeech/LibriSpeech/"
        paths = os.walk(root, topdown=False)
        csv_file = open(args.output + ".csv", "w")
        csv_writer = csv.writer(csv_file, delimiter="\t")
        csv_writer.writerow(["PATH", "SPEAKER", "CHAPTER", "TRANSCRIPT"])
        D = {}
        for i, p in enumerate(paths):
            print("Visiting diretory number {}".format(i))
            # Only consider deepest directories and avoid root
            if p[2] and p[0] != root:
                # Get the transcript
                txt = list(filter(lambda x: x.endswith(".txt"), p[2]))[0]
                trans = open(os.path.join(p[0], txt))
                # Take SPEAKER, CHAPTER and TRANSCRIPT info from the transcript file
                for line in trans:
                    l = line.split()
                    f, t = l[0] + ".flac", " ".join(l[1:])
                    person, chapter, _ = f.split("-")
                    path = os.path.join(p[0], f)
                    arr = extract_mfcc(path)
                    if arr is not None:
                        D[f] = arr
                        csv_writer.writerow([path, person, chapter, t])
                    else:
                        print("NO")
                trans.close()
            if i % 100 == 0 and i != 0:
                print("Saving after {} samples".format(int(i/100)))
                np.save(args.output + "_" + str(int(i/100)), D)
                D = {}
            print(i)
        csv_file.close()
        return
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
    #logging.info("Extracting features from {}".format(f))
    try:
        (sig, rate) = sf.read(f)
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


def get_imgs(dataset):
    if dataset == "places":
        root = "/exp/gchrupal/corpora/placesaudio_distro_part_1"
        img_root = "/roaming/u1257964/Places205/data/vision/torralba/deeplearning/images256"
        names = [ img_root + "/" + line.split()[1] for line in open(root + "/" + "metadata/utt2image") ]
        return names
    elif dataset == "flickr8k":
        raise NotImplementedError
    else:
        raise ValueError ("Unknown dataset {}".format(dataset))

def run_img_feats(args):
    paths = get_imgs(args.dataset)
    feats = img_features(paths, args.cnn, args.resize, args.crop_size, args.tencrop)
    np.save(args.output, feats)

if __name__ == '__main__':
    main()
    
