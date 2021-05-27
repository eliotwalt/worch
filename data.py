import worch
import torch
import matplotlib.pyplot as plt
import math

class Toy():
    def __init__(self, n=1000):
        '''
        Toy class: Generate the data and implements some useful functionalities

        Parameters:
        -----------
        n: int
            Number of datapoints (defaults to 1000)

        Attributes:
        -----------
        X_train: torch.Tensor
            Data points tensor for training

        Y_test: torch.Tensor
            Labels tensor for training

        X_train: torch.Tensor
            Data points tensor for testing

        Y_test: torch.Tensor
            Labels tensor for testing
        '''
        self.n = n
        self.X_train, self.Y_train = self.make()
        self.X_test, self.Y_test = self.make()

    def make(self):
        '''
        Toy.make method: generates data according to toy distribution 

        Parameters:
        -----------
        [No parameter]

        Returns:
        --------
        input: torch.Tensor
            Data points tensor
        
        target: torch.Tensor
            Labels tensor
        '''
        input = torch.empty(self.n, 2).uniform_(0, 1)
        target = (input-torch.Tensor([0.5, 0.5])).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
        target = torch.where(target==0, 1, 0).unsqueeze(-1)
        return input, target

    def onehot(self, X, Y):
        '''
        Toy.onehot method: One-hot encoding of tensor Y

        Parameters:
        -----------
        X: torch.Tensor
            Data points tensor
        
        Y: torch.Tensor
            Labels tensor

        Returns:
        --------
        Yh: torch.Tensor
            Labels tensor, one-hot encoded
        '''
        Yh = X.new_zeros(Y.size(0), Y.max() + 1)
        Yh.scatter_(1, Y.view(-1, 1), 1.0)
        return Yh

    def get(self, test=True, normalize=True, onehot=True):
        '''
        Toy.make method: generates data according to toy distribution 

        Parameters:
        -----------
        test: bool
            if true also returns test set
        
        normalize: bool
            if true data is first normalized
        
        one_hot: bool
            if true result is one hot encoded

        Returns:
        --------
        input: torch.Tensor
            Data points tensor
        
        output: torch.Tensor
            Labels tensor
        '''
        if normalize:
            mu, std = self.X_train.mean(0), self.X_train.std(0)
            self.X_train.sub_(mu).div_(std)
            if test:
                self.X_test.sub_(mu).div_(std)
        if test:
            if onehot:
                return self.X_train, self.onehot(self.X_train, self.Y_train), self.X_test, self.onehot(self.X_test, self.Y_test)
            else:
                return self.X_train, self.Y_train, self.X_test, self.Y_test         
        else:
            if onehot:
                return self.X_train, self.onehot(self.X_train, self.Y_train)
            else:
                return self.X_train, self.Y_train

    def show(self, path=None):
        '''
        Toy.show method: plots the training data

        Parameters:
        -----------
        path: Union[None, str]
            path at which to save the figure. If None then not saved

        Returns:
        --------
        [No output]
        '''
        plt.rcParams['figure.figsize'] = (8,8)
        X0 = self.X_train[(self.Y_train==0).squeeze(1)]
        X1 = self.X_train[(self.Y_train==1).squeeze(1)]
        plt.scatter(X0[:,0].numpy(), X0[:,1].numpy(), c='r', label='1')
        plt.scatter(X1[:,0].numpy(), X1[:,1].numpy(), c='b', label='0')
        plt.legend()
        if path is not None:
            plt.savefig(path)
        plt.show()
        