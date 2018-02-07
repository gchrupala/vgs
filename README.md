# vgs

This is a PyTorch rewrite of  https://github.com/gchrupala/visually-grounded-speech.

- The PyTorch version of the original Theano Recurrent Highway Network-based model is is slow and buggy. 
- The alternative GRU-based model is much faster and more usable.

The data (Flickr8K speech and image features) needed to run the model is here: https://drive.google.com/file/d/14OVoyKibsslVwgYxxgd-s3dbA4bHUZtf/view?usp=sharing
Unpack it in the [data](data) directory.

Then change to [experiments/flickr8k-speech-gru](experiments/flickr8k-speech-gru) and execute:

```
python3 run.py > log.txt
```

The script will run for 25 epochs and print the value of loss function periodically.

