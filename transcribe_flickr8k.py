import argparse
import subprocess
import sys
import glob
import os.path
import scipy.io.wavfile as wav
import numpy as np
from deepspeech.model import Model

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 - '.format(audio_path)
    try:
        p = subprocess.Popen(sox_cmd.split(),
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output, err = p.communicate()

        if p.returncode:
            raise RuntimeError('SoX returned non-zero status: {}'.format(err))

    except OSError as e:
        raise OSError('SoX not found, use 16kHz files or install it: ', e)

    audio = np.fromstring(output, dtype=np.int16)
    return 16000, audio


def main():
    model = "models/output_graph.pb"
    alphabet = "models/alphabet.txt"
    lm = "models/lm.binary"
    trie = "models/trie"

    ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)

    ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

    with open("flickr_audio_transcription.txt", "w") as out:
      for audio_f in glob.glob("/roaming/gchrupal/vgs/data/flickr8k/flickr_audio/wavs/*.wav"):
        print("Transcribing {}".format(audio_f))
        try:
            fs, audio = wav.read(audio_f)
            if fs != 16000:
               if fs < 16000:
                  print('Warning: original sample rate (%d) is lower than 16kHz. Up-sampling might produce erratic speech recognition.' % (fs), file=sys.stderr)
               fs, audio = convert_samplerate(args.audio)
            audio_length = len(audio) * ( 1 / 16000)
            basename, ext = os.path.splitext(os.path.basename(audio_f))
            out.write("{}\t{}\n".format(basename, ds.stt(audio, fs)))
            out.flush()
        except ValueError as e:
            print("Error: {}".format(e))

if __name__ == '__main__':
    main()

