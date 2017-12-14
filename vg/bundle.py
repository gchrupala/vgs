import zipfile
import io
import pickle
import numpy
import torch
import json

class Bundle():

    """Interface for combinations of task/data."""

    def params(self):
        raise NotImplementedError

    def weights(self):
        return [ param.data.cpu().numpy() for param in self.params() ]

    def get_config(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedEr
    def save(self, path):
        zf = zipfile.ZipFile(path, 'w')
        buf = io.BytesIO()
        numpy.save(buf, self.weights())
        zf.writestr('weights.npy', buf.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('config.json', json.dumps(self.get_config()),
                    compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr('data.pkl', pickle.dumps(self.get_data()),
                    compress_type=zipfile.ZIP_DEFLATED)

class GenericBundle(Bundle):
    """Generic subclass of Bundle which stores common types of settings."""
    def __init__(self, data, config, task, weights=None):
        self.config = config
        self.config['task'] = list(pickle.dumps(task))
        self.data = data
        self.batcher = data['batcher']
        self.scaler = data['scaler']
        if config.get('size_vocab') is None:
            self.config['size_vocab'] = self.data['batcher'].mapper.size()
        else:
            self.config['size_vocab'] = config['size_vocab']
        self.task = task(config)
        if weights is not None:
            assert len(self.task.parameters())==len(weights)
            for param, weight in zip(self.params(), weights):
                param.data = torch.from_numpy(weight)
#        self.task.compile()
#        self.task.representation = self.task._make_representation()
#        self.task.pile = self.task._make_pile()

    def params(self):
        return self.task.parameters()

    def get_config(self):
        return self.config

    def get_data(self):
        return self.data


# The following functions work on GenericBundle

def load(path):
    """Load data and reconstruct model."""
    with zipfile.ZipFile(path,'r') as zf:
        buf = io.BytesIO(zf.read('weights.npy'))
        weights = numpy.load(buf)
        config  = json.loads(zf.read('config.json'))
        data  = pickle.loads(zf.read('data.pkl'))
        task = pickle.loads(b''.join(config['task']))
    return GenericBundle(data, config, task, weights=weights)
