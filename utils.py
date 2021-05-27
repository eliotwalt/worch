import torch
import worch
torch.set_grad_enabled(False)

def train_worch(model, x, y, criterion, optimizer):
    '''
    Training function for worch model
    
    Parameters:
    -----------
    model: worch.nn.Module
        Model
    
    x, y: torch.Tensor
        Input, target tensors
    
    criterion: worch.nn.Module
        Loss

    optimizer: worch.optim.Optimizer
        optimizer

    Returns:
    --------
    loss: float
        Value of the loss

    yp: torch.Tensor
        prediction
    '''
    model.train()
    yp = model(x)
    loss = criterion(yp, y)
    optimizer.zero_grad()
    criterion.backward() # call on module not tensor
    optimizer.step()
    return loss.cpu().detach().item(), yp

def train_torch(model, x, y, criterion, optimizer):
    '''
    Training function for torch model
    
    Parameters:
    -----------
    model: torch.nn.Module
        Model
    
    x, y: torch.Tensor
        Input, target tensors
    
    criterion: torch.nn.Module
        Loss

    optimizer: torch.optim.Optimizer
        optimizer

    Returns:
    --------
    loss: float
        Value of the loss

    yp: torch.Tensor
        prediction
    '''
    model.train()
    torch.set_grad_enabled(True)
    yp = model(x)
    loss = criterion(yp, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.set_grad_enabled(False)
    return loss.cpu().detach().item(), yp

def eval_model(model, x, y, criterion):
    '''
    Evaluation function for torch or worch model
    
    Parameters:
    -----------
    model: torch.nn.Module
        Model
    
    x, y: torch.Tensor
        Input, target tensors
    
    criterion: torch.nn.Module or worch.nn.Module
        Loss

    Returns:
    --------
    loss: float
        Value of the loss
    '''
    model.eval()
    yp = model(x)
    loss = criterion(yp, y)
    return loss.cpu().detach().item()

def binary_num_errors(model, x, y):
    '''
    Compute binary classification number of errors of torch or worch model
    
    Parameters:
    -----------
    model: worch.nn.Module
        Model
    
    x, y: torch.Tensor
        Input, target tensors

    Returns:
    --------
    e: float
        Number of errors
    '''
    yp = model(x)
    preds = torch.argmax(yp, dim=1)
    gt = torch.argmax(y, dim=1)
    e = (preds!=gt).long().sum()
    return e