# Worch

Worch is an implementation of a simple deep learning library based on pytorch. It supports MLPs with tanh, sigmoid or relu activations, MSE loss and SGD optimizer. It's been developed as a project for the **EE-559 Deep Learning** class at EPFL. The main goal was to implement backpropagation from scratch, i.e without using `torch.autograd`.

## Design

The design is very similar to pytorch. The neural networks components and loss functions inherit from `worch.nn.Module` and the optimizers from `worch.optim.Optimizer`. Implementing a new module, however, requires to explicitly implement the backward pass in the `backward` method, as `torch.autograd` is not active.

## Example

Let's create a MLP with MSE loss and SGD optimizer using worch and train it:

```python
import worch
from data import Toy

# param
batch_size = 128
n_epochs = 20

# Get some data
toy = Toy()
X_train, Y_train = toy.get(test=False, normalize=True, onehot=True)

# Build sequential model
net = worch.nn.Sequential(
    worch.nn.Linear(in_features=X_train.shape[1], out_features=25),
    worch.nn.ReLU(),
    worch.nn.Linear(in_features=25, out_features=16),
    worch.nn.ReLU(),
    worch.nn.Linear(in_features=16, out_features=2),
    worch.nn.Sigmoid()
)

# Build loss
crit = worch.nn.LossMSE()
crit.register_previous_module(net) # Required additional boilerplate

# Build optimizer
optimizer = worch.optim.SGD(net.parameters(), lr=0.1)

# training loop
for epoch in range(n_epochs):
    for b in range(0, X_train.shape[0], batch_size):
        x = X_train[b:b+batch_size]
        y = Y_train[b:b+batch_size]    
        # forward pass
        yp = net(x)    
        # Compute loss
        loss = crit(yp, y)    
        # Backprop
        optimizer.zero_grad() # Reset gradients
        crit.backward() # Compute gradients: call on module, not `loss` tensor
        optimizer.step() # Compute gradient step
```

