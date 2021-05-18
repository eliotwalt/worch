import torch
import math
from sklearn.datasets import make_friedman1

def generate_data(n, kind):
    if kind == 'disk':
        input = torch.empty(n, 2).uniform_(0, 1)
        target = (input-torch.Tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
        target = torch.where(target==0, 1, 0).unsqueeze(-1)
    elif kind == 'sin':
        input = torch.arange(n).float().unsqueeze(-1)
        target = torch.sin(input)+torch.ones_like(input).float().normal_(0,0.1)
    elif kind == 'friedman':
        input, target = make_friedman1(n)
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).unsqueeze(-1).float()
    return input, target
