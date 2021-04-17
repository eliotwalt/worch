import torch
torch.set_grad_enabled(False)

class Optimizer(object):

    '''
    Parent class to inherit from
    '''

    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError('No implementation of step found')

    def zero_grad(self):
        for w in self.parameters:
            w.grad = torch.zeros_like(w)

class SGD(Optimizer):

    def __init__(self, parameters, lr):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for w in self.parameters:
            w -= self.lr * w.grad
