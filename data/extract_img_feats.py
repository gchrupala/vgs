import sys

#TODO not the most elegant solution
# Need to clone https://github.com/GKalliatakis/Keras-VGG16-places365
sys.path.append("/home/u1257964/Keras-VGG16-places365/")
from keras.models import Model
from vgg16_hybrid_places_1365 import VGG16_Hubrid_1365
from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from places_utils import preprocess_input as preprocess_vgg16hybrid
from PIL import Image, ImageOps
import numpy as np


def resize_image(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size. From torchvision

        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop_image(image, crop_size=224, pos="c", hflip=False):
    """Crop image.
    
    pos: str, [c, tl, bl, tr, br]
               (c)enter, (t)op, (b)ottom, (l)eft, (r)ight
    """
    if hflip:
        image = ImageOps.mirror(image)
    w, h = image.size
    boxes = {"tl": (0, 0, crop_size, crop_size),
             "tr": (w-crop_size, 0, w, h-(h-crop_size)),
             "br": (0, h-crop_size, crop_size, h), 
             "bl": (w-crop_size, h-crop_size, w, h), 
             "c": (w/2 - crop_size/2, h/2 - crop_size/2,
                   w/2 + crop_size/2, h/2 + crop_size/2)}
    image = image.crop(boxes[pos])
    return image


def img_features(paths, cnn="vgg16", resize=None, crop_size=224, tencrop=False):
    """
    Slow, but simple batchsize 1 image extractor.
    
    paths: str list, list of image paths
    cnn: str, which pre-trained cnn to use
    resize: int or tuple, if int then rescale such that smaller side is resize length
                          if tuple then just interpolate to target size.
    crop_size: int, size of the square cropped from the resized image
    """
    if tencrop and not crop_size:
        print("If tencrop is True please provide crop size")
        raise NameError
    out_dim = 4096
    FEATS = np.zeros((len(paths), out_dim))
    if cnn == "vgg16":
        base_model = VGG16(weights="imagenet")
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)        
        preprocess_input = preprocess_vgg16
    elif cnn == "vgg19":
        base_model = VGG19(weights="imagenet")
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)        
        preprocess_input = preprocess_vgg19
    # Run VGG trained on both Places365 and ImageNet
    #TODO Doesnt work :D i think because of keras version compat
    elif cnn == "hybrid":
        base_model = VGG16_Hubrid_1365(weights='places')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)        
        preprocess_input = preprocess_vgg16hybrid
    else:
        raise NotImplementedError
    model.trainable = False  # Saving space
    for i, img_path in enumerate(paths):
        # Read image and resize with bilinear interpolation if given
        image = keras_image.load_img(img_path)
        if resize:
            image = resize_image(image)
        # Take center crop and forward pass
        if crop_size and not tencrop:
            image = crop_image(image, crop_size=crop_size, pos="c", hflip=False)
            x = keras_image.img_to_array(image)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x).squeeze()
        # Apply 10crop strategy from order-embeddings https://arxiv.org/abs/1511.06361
        elif tencrop:
            X = []
            for f in [True, False]:
                for p in ["c", "tl", "bl", "tr", "br"]:
                    img = crop_image(image, crop_size=crop_size, pos=p, hflip=f)
                    x = keras_image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    feat = model.predict(x).squeeze()
                    X.append(feat)
            # Take average mean over all crops
            features = np.array(X)
            features = np.mean(features, axis=0)
        FEATS[i] = features
    return FEATS
