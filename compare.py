import torch
import worch
import matplotlib.pyplot as plt
from data import Toy
from utils import *
torch.set_grad_enabled(False)

class Options:
    def __init__(self):
        self.n_epochs = 100
        self.batch_size = 32
        self.lr = 0.1
        self.hidden_size = 25

def main():
    # Get parameters
    opt = Options()
    # Get data
    toy = Toy()
    X_train, Y_train, X_test, Y_test = toy.get(test=True, normalize=True, onehot=True)
    # Build model
    wnet = worch.nn.Sequential(
        worch.nn.Linear(X_train.shape[1], opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, 2),
        worch.nn.Sigmoid()
    )
    tnet = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], opt.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(opt.hidden_size, opt.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(opt.hidden_size, opt.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(opt.hidden_size, 2),
        torch.nn.Sigmoid()
    )
    # Build loss and optimizer
    wcrit = worch.nn.LossMSE()
    wcrit.register_previous_module(wnet)
    tcrit = torch.nn.MSELoss()
    woptim = worch.optim.SGD(wnet.parameters(), lr=opt.lr)
    toptim = torch.optim.SGD(tnet.parameters(), lr=opt.lr)
        # training loop
    w_trlosses = []
    w_valosses = []
    t_trlosses = []
    t_valosses = []
    for epoch in range(1, opt.n_epochs+1):
        # Train
        wnet.train()
        tnet.train()
        wlosses = []
        tlosses = []
        for k, b in enumerate(range(0, X_train.shape[0], opt.batch_size)):
            x = X_train[b:b+opt.batch_size]
            y = Y_train[b:b+opt.batch_size]
            wloss, wyp = train_worch(wnet, x.clone(), 
                                    y.clone(), wcrit, 
                                    woptim)
            tloss, typ = train_torch(tnet, x.clone(), 
                                    y.clone(), tcrit, 
                                    toptim)
            wlosses.append(wloss)
            tlosses.append(tloss)
        w_trlosses.append(torch.mean(torch.Tensor(wlosses)))
        t_trlosses.append(torch.mean(torch.Tensor(tlosses)))
        # Val
        wnet.eval()
        tnet.eval()
        wlosses = []
        tlosses = []
        for b in range(0, X_test.shape[0], opt.batch_size+1):
            x = X_test[b:b+opt.batch_size+1]
            y = Y_test[b:b+opt.batch_size+1]
            wloss = eval_model(wnet, x, y, wcrit)
            tloss = eval_model(tnet, x, y, tcrit)
            wlosses.append(wloss)
            tlosses.append(tloss)
        w_valosses.append(torch.mean(torch.Tensor(wlosses)))
        t_valosses.append(torch.mean(torch.Tensor(tlosses)))
        if epoch==1 or epoch % 10 == 0:
            print('[torch@({}/{})] train: {}, val: {}'.format(
                epoch, opt.n_epochs, t_trlosses[-1], t_valosses[-1]
            ))
            print('[worch@({}/{})] train: {}, val: {}\n-'.format(
                epoch, opt.n_epochs, w_trlosses[-1], w_valosses[-1]
            ))
    # Show loss
    plt.rcParams['figure.figsize'] = (5,3)
    n = torch.arange(opt.n_epochs)
    plt.plot(n, w_trlosses, 'r', label='worch train')
    plt.plot(n, w_valosses, 'k', label='worch val')
    plt.plot(n, t_trlosses, 'g', label='torch train')
    plt.plot(n, t_valosses, 'b', label='torch val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
