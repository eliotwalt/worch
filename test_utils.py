import torch
import math

def generate_data(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input-torch.Tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    target = torch.where(target==0, 1, 0).unsqueeze(-1)
    return input, target