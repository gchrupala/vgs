import numpy as np
import torch

def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape).astype('float32')
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return torch.from_numpy(scale * q[:shape[0], :shape[1]])

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)

def xavier(shape):
    nin, nout = shape
    r = np.sqrt(6.) / np.sqrt(nin + nout)
    W = np.random.rand(nin, nout).astype('float32') * 2 * r - r
    return torch.from_numpy(W)

# https://github.com/fchollet/keras/blob/master/keras/initializations.py
def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def uniform(shape, scale=0.05):
    return torch.from_numpy(np.random.uniform(low=-scale, high=scale, size=shape).astype('float32'))
    
