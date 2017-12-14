import torch.nn as nn
import onion.util as util
import onion.init as init

class Convolution1D(nn.Module):
    """A one-dimensional convolutional layer.
    """
    def __init__(self, size_in, length, size, stride=1):
        super(Convolution1D, self).__init__()
        util.autoassign(locals())
        self.Conv = nn.Conv1d(self.size_in, self.size, self.length, stride=self.stride, padding=self.length, bias=False)
        # use Glorot uniform initialization
        self.Conv.weight.data = init.glorot_uniform((self.size, self.size_in, self.length, 1)).squeeze()
        #FIXME what is the correct padding???

    def forward(self, signal):
        # signal's shape is (B, T, C) where B=batch size, T=timesteps, C=channels
        out = self.Conv(signal.permute(0, 2, 1))
        return out.permute(0, 2, 1)
