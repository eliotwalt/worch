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
    net = worch.nn.Sequential(
        worch.nn.Linear(X_train.shape[1], opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, opt.hidden_size),
        worch.nn.ReLU(),
        worch.nn.Linear(opt.hidden_size, 2),
        worch.nn.Sigmoid()
    )
    # Build loss and optimizer
    crit = worch.nn.LossMSE()
    crit.register_previous_module(net)
    optim = worch.optim.SGD(net.parameters(), lr=opt.lr)
    # training loop
    trlosses = []
    trerrs = []
    vallosses =[]
    valerrs = []
    for epoch in range(1, opt.n_epochs+1):
        # Train
        net.train()
        losses = []
        errs = []
        for k, b in enumerate(range(0, X_train.shape[0], opt.batch_size)):
            x = X_train[b:b+opt.batch_size]
            y = Y_train[b:b+opt.batch_size]
            loss, _ = train_worch(net, x, y, crit, optim)
            nerr = binary_num_errors(net, x, y)
            losses.append(loss)
            errs.append(nerr)
        trlosses.append(torch.mean(torch.Tensor(losses)))
        trerrs.append(torch.sum(torch.Tensor(errs)))
        # Val
        net.eval()
        losses = []
        errs = []
        for b in range(0, X_test.shape[0], opt.batch_size+1):
            x = X_test[b:b+opt.batch_size+1]
            y = Y_test[b:b+opt.batch_size+1]
            loss = eval_model(net, x, y, crit)
            nerr = binary_num_errors(net, x, y)
            losses.append(loss)
            errs.append(nerr)
        vallosses.append(torch.mean(torch.Tensor(losses)))
        valerrs.append(torch.sum(torch.Tensor(errs)))
        if epoch==1 or epoch % 10 == 0:
            print('[worch@({}/{})] train loss: {}, val loss: {}\n-'.format(
                epoch, opt.n_epochs, trlosses[-1], vallosses[-1], trerrs[-1], valerrs[-1]
            ))
    print('\nFINAL ERROR RATE:\ntrain: {}\nvalidation:{}'.format(trerrs[-1], valerrs[-1]))
    # Show loss
    plt.rcParams['figure.figsize'] = (8,5)
    n = torch.arange(opt.n_epochs)
    plt.plot(n, trlosses, 'r', label='worch train')
    plt.plot(n, vallosses, 'k-', label='worch val')
    plt.legend()    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('worch_losses.png')
    plt.show()
    # Show errors
    n = torch.arange(opt.n_epochs)
    plt.plot(n, trerrs, 'r', label='worch train')
    plt.plot(n, valerrs, 'k-', label='worch val')
    plt.xlabel('epoch')
    plt.ylabel('number of error')
    plt.legend()
    plt.savefig('worch_nerr.png')
    plt.show()

if __name__ == '__main__':
    main()
