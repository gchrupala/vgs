import torch

def parse_map(lines):
    M = {}
    for line in lines:
        fields  =line.split()
        M[fields[0]] = ' '.join(fields[1:])
    return M

def contrastive(M, margin=0.2):            
     E = - M
     D = torch.diag(E)
     C_c = torch.clamp(margin - E + D, min=0)
     C_r = torch.clamp(margin - E + D.view(-1,1), min=0)
     C = C_c + C_r
     return (C.sum() - torch.diag(C).sum())/C.size(0)**2

