import numpy as np

def length_out(L, size, stride, padding, dilation=1):
    "Return the length of the output given the length of the input and the parameters of 
    the 1D convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d"

    return np.floor(  (L + 2 * padding - dilation * (size-1) - 1) / 
                       stride +
                      1
                    )
                        

