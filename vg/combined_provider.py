import itertools
import numpy as np
import vg.places_provider as places
import vg.flickr8k_provider as flickr
import vg.vendrov_provider as vendrov

class CombinedProvider:
    def __init__(self, providers):
        self.providers = providers

    def iterSentences(self, split = 'train'):
        iters = [ p.iterSentences(split=split) for p in self.providers ]
        for item in itertools.chain(*iters):
          yield item

    def iterImages(self, split = 'train'):
        iters = [ p.iterImages(split=split) for p in self.providers ]
        for item in itertools.chain(*iters):
          yield item

class PlacesFlickr8KProvider:
   
    def __init__(self, root='.', truncate=None):
        self.places = places.getDataProvider('places', root=root, truncate=truncate)
        self.flickr = flickr.getDataProvider('flickr8k', root=root)

    def iterSentences(self, split = 'train', shuffle=False):
        if split == 'train':
            it = itertools.chain(self.places.iterSentences(split=split), self.flickr.iterSentences(split=split))
        else:
            it = self.flickr.iterSentences(split=split)
        if shuffle:
            yield from sorted(it, key=lambda _: np.random.random())
        else:
            yield from it

    def iterImages(self, split = 'train', shuffle=False):
        if split == 'train':
            it = itertools.chain(self.places.iterImages(split=split), self.flickr.iterImages(split=split))
        else:
            it = self.flickr.iterImages(split=split)
        #it = itertools.chain(self.places.iterImages(split=split), self.flickr.iterImages(split=split))
        if shuffle:
            yield from sorted(it, key=lambda _: np.random.random())
        else:
            yield from it

class PlacesCocoProvider:
    
    def __init__(self, root='.', truncate=None, load_images=True):
        self.places = places.getDataProvider('places', root=root, truncate=truncate, load_images=load_images)
        self.coco = vendrov.getDataProvider('coco', root=root, truncate=truncate, load_images=load_images)
        self.speakers = self.places.speakers.union(self.coco.speakers)
    
    def iterSentences(self, split = 'train', shuffle=False):
        it = itertools.chain(self.places.iterSentences(split=split), self.coco.iterSentences(split=split))
        if shuffle:
            yield from sorted(it, key=lambda _: np.random.random())
        else:
            yield from it

    def iterImages(self, split = 'train', shuffle=False):
        it = itertools.chain(self.places.iterImages(split=split), self.coco.iterImages(split=split))
        if shuffle:
            yield from sorted(it, key=lambda _: np.random.random())
        else:
            yield from it


