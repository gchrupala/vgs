import itertools
import onion.init as init
import torch.nn as nn
import torch

def autoassign(locs):
    """Assign locals to self."""
    for key in locs.keys():
        if key!="self":
            locs["self"].__dict__[key]=locs[key]

def grouper(iterable, n):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        chunks = itertools.zip_longest(fillvalue=None, *args)
        for chunk in chunks:
            yield [ x for x in chunk if not x is None ]

def make_linear(size_in, size_out):
    """Returns linear layer with orthogonal initialization."""
    M = nn.Linear(size_in, size_out)
    M.weight.data = init.orthogonal((size_out, size_in))
    M.bias.data = torch.zeros(size_out)
    return M
