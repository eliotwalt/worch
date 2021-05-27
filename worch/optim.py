import torch
torch.set_grad_enabled(False)

class Optimizer(object):

    def __init__(self, parameters):
        '''
        worch.optim.Optimizer class: used as a base class to all worch optimizers

        Parameters:
        -----------
        parameters: list[torch.Tensor]
            List of parameters to optimize

        Attributes:
        -----------
        parameters: list[torch.Tensor]
            List of parameters to optimize
        '''
        self.parameters = parameters

    def step(self):
        '''
        worch.optim.Optimizer.step: perform an update step

        Parameters:
        -----------
        [No parameters]

        Errors:
        -------
        NotImplementedError
            Not implementd for base class worch.optim.Optimizer

        Returns:
        --------
        [No output]
        '''
        raise NotImplementedError('No implementation of step found')

    def zero_grad(self):
        '''
        worch.optim.Optimizer.zero_grad: reset gradient field of parameters to zero

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        [No output]
        '''
        for w in self.parameters:
            w.grad = None

class SGD(Optimizer):

    def __init__(self, parameters, lr):
        '''
        worch.optim.SGD class: used as a base class to all worch optimizers
        (inherited from worch.optim.Optimizer)

        Parameters:
        -----------
        parameters: list[torch.Tensor]
            List of parameters to optimize
        lr: float
            learning rate

        Attributes:
        -----------
        parameters: list[torch.Tensor]
            List of parameters to optimize
        lr: float
            learning rate
        '''
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        '''
        worch.optim.SGD.step: perform an SGD update step

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        [No output]
        '''
        for w in self.parameters:
            w -= self.lr * w.grad
