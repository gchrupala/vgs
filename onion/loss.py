import torch
import torch.nn.functional as F

def contrastive(M, margin=0.2):       
     "Returns contrastive margin loss over similarity matrix M."     
     E = - M
     D = torch.diag(E)
     C_c = torch.clamp(margin - E + D, min=0)
     C_r = torch.clamp(margin - E + D.view(-1,1), min=0)
     C = C_c + C_r
     return (C.sum() - torch.diag(C).sum())/C.size(0)**2

def rsa(A, B):
    "Returns the correlation between the similarity matrices for A and B."
    M_A = cosine_matrix(A, A)
    M_B = cosine_matrix(B, B)
    return pearson(triu(M_A), triu(M_B), dim=0) 

def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())


def pearson(x, y, dim=1, eps=1e-8):
    "Returns Pearson's correlation coefficient."
    x1 = x - torch.mean(x, dim)
    x2 = y - torch.mean(y, dim)
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)

def triu(x):
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = torch.autograd.Variable(torch.ones_like(x.data))
    return x[torch.triu(ones, diagonal=1) == 1]


