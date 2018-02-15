import sys
import os
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
from torchvision.transforms import functional as F
import numpy as np
from tqdm import tqdm


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
    FEATS = np.zeros((len(paths), out_dim), dtype="float32")
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
    for i, img_path in tqdm(list(enumerate(paths))):
        # Read image and resize with bilinear interpolation if given
        image = keras_image.load_img(img_path)
        if resize:
            image = F.resize(image, resize)
        # Take center crop and forward pass
        if crop_size and not tencrop:
            image = F.center_crop(image, crop_size)
            x = keras_image.img_to_array(image)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x).squeeze()
        # Apply 10crop strategy from order-embeddings https://arxiv.org/abs/1511.06361
        elif tencrop:
            X = []
            imgs = F.ten_crop(image, crop_size)
            for img in imgs:
                x = keras_image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feat = model.predict(x).squeeze()
                X.append(feat)
            # Take average mean over all crops
            features = np.array(X)
            features = np.mean(features, axis=0)
        # No cropping case
        else:
            x = keras_image.img_to_array(image)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x).squeeze()

        FEATS[i] = features
        image.close()
    return FEATS

def test():
    root = "/roaming/u1257964/Places205/data/vision/torralba/deeplearning/images256/a/abbey/"
    paths = os.listdir("/roaming/u1257964/Places205/data/vision/torralba/deeplearning/images256/a/abbey/")
    paths = list(map(lambda x: os.path.join(root, x), paths))
    m = img_features(paths, crop_size=224, cnn="vgg16")
    print(m)

if __name__ == "__main__":
    test()
