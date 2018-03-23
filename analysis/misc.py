import torch
seed = 666
torch.cuda.manual_seed_all(seed)
import torch.autograd as autograd

class MeanNet:
    def __init__(self):
        self.training = False
    def eval(self):
        pass
    
    def predict(self, x):
        return x.mean(dim=1)
    
class RandNet:
    def __init__(self, features=1024):
        self.training = False
        self.features = features
    
    def eval(self):
        pass
        
    def predict(self, x):
        return autograd.Variable(torch.FloatTensor(x.size(0), self.features).uniform_(-1, 1))