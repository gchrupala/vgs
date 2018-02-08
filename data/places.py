import os

root = "/exp/gchrupal/corpora/placesaudio_distro_part_1/"
utt2wav = open(os.path.join(os.path.join(root, "metadata"), "utt2wav")).read().split("\n")[:-1]
utt2image = open(os.path.join(os.path.join(root, "metadata"), "utt2image")).read().split("\n")[:-1]
kaldi2wav = dict(map(lambda x: x.split(), utt2wav))
kaldi2img = dict(map(lambda x: x.split(), utt2image))
wav2img = {}
for k in kaldi2wav:
    wav = kaldi2wav[k]
    wav2img[wav] = kaldi2img[k]
