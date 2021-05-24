import torch
torch.set_grad_enabled(False)

class Module(object):

    def __init__(self):
        '''
        worch.nn.Module class: used as a base class to all worch modules

        Parameters:
        -----------
        [No parameters]

        Attributes:
        -----------
        last_input: Union[None, torch.Tensor]
            Records the last input tensor feeded to the module. initialized to None

        previous_module: Union[None, worch.nn.Module]
            Points to previous module in the computation graph
        
        params: List[torch.Tensor]
            List of module parameters to optimize

        training: bool
            boolean field indicating if the model is in training mode

        device: torch.device
            [Not implemented] Device on which to compute
        '''
        self.last_input = None
        self.previous_module = None
        self.params = []
        self.training = True
        self.device = torch.device('cpu')

    def forward(self, *input):
        '''
        worch.nn.Module.forward method: forward pass

        Parameters:
        -----------
        *input: tuple
            Module's forward pass input

        Errors:
        -------
        NotImplementedError
            Not implementd for base class worch.nn.Module
        '''
        raise NotImplementedError('No implementation of forward found')

    def __call__(self, *input):
        '''
        worch.nn.Module.__call__ method: forward pass

        Parameters:
        -----------
        *input: tuple
            Module's forward pass input

        Returns:
        --------
        self.forward(*input)
            Result of forward pass
        '''
        return self.forward(*input)
    
    def backward(self, *gradwrtoutput):
        '''
        worch.nn.Module.backward method: backward pass

        Parameters:
        -----------
        *gradwrtouput: tuple
            Gradient(s) with respect to output(s) of the Module

        Errors:
        -------
        NotImplementedError
            Not implementd for base class worch.nn.Module

        Returns:
        --------
        [No output]
        '''
        raise NotImplementedError('No implementation of backward found')

    def initialize_parameters(self):
        '''
        worch.nn.Module.initialize_parameters method: initialize parameters

        Parameters:
        -----------
        [No parameters]

        Errors:
        -------
        NotImplementedError
            Not implementd for base class worch.nn.Module

        Returns:
        --------
        [No output]
        '''
        raise NotImplementedError('No implementation of initialize_parameters found')

    def to(self, device):
        '''
        worch.nn.Module.to method: switch to another computing device

        [NOT IMPLEMENTED]

        Parameters:
        -----------
        device: torch.device
            Device to switch to

        Returns:
        --------
        self: worch.nn.Module
            Self reference to Module
        '''
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        for i in range(len(self.params)):
            if len(self.params[i]) > 0:
                self.params[i] = self.params[i].to(device)
        return self

    def parameters(self):
        '''
        worch.nn.Module.parameters method: get list of parameters 

        [NOT IMPLEMENTED]

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        self.params: List[torch.Tensor]
            List of module parameters to optimize
        '''
        return self.params

    def register_previous_module(self, module):
        '''
        worch.nn.Module.register_previous_module method: register previous module in computation graph

        Parameters:
        -----------
        module: worch.nn.Module
            Module to register

        Errors:
        -------
        AssertionError
            Passed module is not an instance of worch.nn.Module

        Returns:
        --------
        [No output]
        '''
        assert isinstance(module, Module), f'module must be Module.'
        self.previous_module = module

    def propagate_gradient(self, gradwrtinput):
        '''
        worch.nn.Module.propagate_gradient method: propagates gradient to previous module in the graph or returns if root

        Parameters:
        -----------
        gradwrtinput: torch.Tensor
            gradient with respect to the input of the module

        Returns:
        --------
        if root:
            gradwrtinput: torch.Tensor
        else:
            self.previous_module.backward(gradwrtinput): propagates in the graph backward
        '''
        if self.previous_module is None:
            return gradwrtinput
        else:
            return self.previous_module.backward(gradwrtinput)

    def train(self):
        '''
        worch.nn.Module.train method: set to training mode

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        [No output]
        '''
        self.training = True
    
    def eval(self):
        '''
        worch.nn.Module.eval method: set to evaluation (i.e not triaining) mode

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        [No output]
        '''
        self.training = False

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        '''
        worch.nn.Linear class: Fully connected module
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        in_features: int
            Number of input features

        out_features: int
            Number of output features

        bias: bool
            Add bias or not
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        # Params is [W, (b)]
        self.params.append(torch.zeros((self.out_features, self.in_features)))
        if self.use_bias:
            bias = torch.zeros((self.out_features, 1))
            self.params.append(bias)

        self.initialize_parameters()

    def forward(self, *input):
        '''
        worch.nn.Linear.forward method: forward pass

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        y: torch.Tensor
            Linear output
        '''
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        # print('x',x.shape)
        y = torch.mm(x, self.params[0].T)
        # print('wT',self.params[0].T.shape)
        if self.use_bias:
            y = y + self.params[1]
        # print('y',y.shape)
        # print()
        return y

    def backward(self, *gradwrtoutput):
        '''
        worch.nn.Linear.backward method: backward pass. Computes gradient wrt to input and parameters.

        Parameters:
        -----------
        *gradwrtoutput: tuple
            Gradient(s) with respect to output(s) of the Module

        Returns:
        --------
        self.propagate_gradient(gradwrtinput)
        '''
        gradwrtoutput, = gradwrtoutput
        # Compute gradient wrt to parameters and store in grad field of tensors
        w_gradient = torch.mm(gradwrtoutput.T, self.last_input)
        if self.params[0].grad is None:
            self.params[0].grad = w_gradient
        else:
            self.params[0].grad += w_gradient
        if self.use_bias:
            b_gradient = torch.sum(gradwrtoutput, axis=0)
            # print(self.params[1].shape, b_gradient.shape)
            if self.params[1].grad is None:
                self.params[1].grad = b_gradient
            else:
                self.params[1].grad += b_gradient
        # print('--------------------------------------------')
        # print('linear gradwrtoutput: ',gradwrtoutput.shape)
        # print('linear lastinput: ',self.last_input.shape)
        # print('linear W: ',self.params[0].shape)
        # print('')
        gradwrtinput = torch.mm(gradwrtoutput, self.params[0])
        # print('linear gradwrtinput: ', gradwrtinput.shape)
        # print('--------------------------------------------')
        self.last_input = None
        
        return self.propagate_gradient(gradwrtinput)

    def initialize_parameters(self):
        '''
        worch.nn.Module.initialize_parameters method: initialize parameters with xavier_uniform

        Parameters:
        -----------
        [No parameters]

        Returns:
        --------
        [No output]
        '''
        self.params[0] = torch.nn.init.xavier_uniform_(self.params[0])
        self.params[1] = torch.nn.init.xavier_uniform_(self.params[1]).squeeze(-1)

class ReLU(Module):

    def __init__(self):
        '''
        worch.nn.ReLU class: ReLU activation
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        [No parameters]
        '''
        super().__init__()

    def forward(self, *input):
        '''
        worch.nn.ReLU.forward method: forward pass

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        y: torch.Tensor
            Rectified output
        '''
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        y = torch.clamp(x, min=0)
        return y

    def backward(self, *gradwrtoutput):
        '''
        worch.nn.ReLU.backward method: backward pass. Computes gradient wrt to input.

        Parameters:
        -----------
        *gradwrtoutput: tuple
            Gradient(s) with respect to output(s) of the Module

        Returns:
        --------
        self.propagate_gradient(gradwrtinput)
        '''
        gradwrtoutput, = gradwrtoutput
        # print('--------------------------------------------')
        # print('relu gradwrtoutput: ',gradwrtoutput.shape)
        # print('relu lastinput: ',self.last_input.shape)
        gradwrtinput = torch.ones_like(self.last_input)*gradwrtoutput
        gradwrtinput[self.last_input < 0] = 0
        # print('relu gradwrtinput: ', gradwrtinput.shape)
        # print('--------------------------------------------')
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class Sigmoid(Module):

    def __init__(self):
        '''
        worch.nn.Sigmoid class: Sigmoid activation
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        [No parameters]
        '''
        super().__init__()

    def forward(self, *input):
        '''
        worch.nn.Sigmoid.forward method: forward pass

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        y: torch.Tensor
            Sigmoidal output
        '''
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        mask = torch.zeros_like(x)
        positives = torch.where(x>=0, x, mask)
        negatives = torch.where(x<0, x, mask)
        pos_idx = x>=0
        neg_idx = x<0
        # Avoid overflow
        exp_pos = torch.exp(-positives)
        exp_pos[neg_idx] = 0
        exp_neg = torch.exp(negatives)
        exp_neg[pos_idx] = 0
        sigmoid_pos = 1/(1+exp_pos)
        sigmoid_pos[neg_idx] = 0
        sigmoid_neg = exp_neg/(1+exp_neg)
        sigmoid_neg[pos_idx] = 0
        return sigmoid_pos+sigmoid_neg

    def backward(self, *gradwrtoutput):
        '''
        worch.nn.Sigmoid.backward method: backward pass. Computes gradient wrt to input.

        Parameters:
        -----------
        *gradwrtoutput: tuple
            Gradient(s) with respect to output(s) of the Module

        Returns:
        --------
        self.propagate_gradient(gradwrtinput)
        '''
        gradwrtoutput, = gradwrtoutput
        output_sigmoid = self.forward(self.last_input)
        gradwrtinput = (output_sigmoid*(1-output_sigmoid))*gradwrtoutput
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class Tanh(Module):

    def __init__(self):
        '''
        worch.nn.Tanh class: Tanh activation
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        [No parameters]
        '''
        super().__init__()

    def forward(self, *input):
        '''
        worch.nn.Tanh.forward method: forward pass

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        y: torch.Tensor
            Tanh output
        '''
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        return (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))

    def backward(self, *gradwrtoutput):
        '''
        worch.nn.Tanh.backward method: backward pass. Computes gradient wrt to input.

        Parameters:
        -----------
        *gradwrtoutput: tuple
            Gradient(s) with respect to output(s) of the Module

        Returns:
        --------
        self.propagate_gradient(gradwrtinput)
        '''
        gradwrtoutput, = gradwrtoutput
        output_tanh = self.forward(self.last_input)
        gradwrtinput = (1-torch.pow(output_tanh, 2))*gradwrtoutput
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class MSELoss(Module):

    def __init__(self):
        '''
        worch.nn.MSELoss class: MSELoss activation
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        [No parameters]
        '''
        super().__init__()

    def forward(self, input, target):
        '''
        worch.nn.MSELoss.forward method: forward pass

        Parameters:
        -----------
        input: torch.Tensor
            Tensor of prediction to compute the error of
        target: torch.Tensor
            Tensor of ground truth to compute the error from

        Returns:
        --------
        y: torch.Tensor
            output
        '''
        target = target.reshape(input.shape) # just in case
        if self.training:
            self.last_input = [input, target]
        agg = torch.pow(input-target, 2)
        return torch.mean(agg)

    def backward(self):
        '''
        worch.nn.Tanh.backward method: backward pass. Computes gradient wrt to input.

        Parameters:
        -----------
        [No parameters] (loss function => no gradwrtoutput)

        Returns:
        --------
        self.propagate_gradient(gradwrtinput)
        '''
        gradwrtinput = (self.last_input[0]-self.last_input[1])*2/30
        # print('--------------------------------------------')
        # print('mse lastinput[input]: ',self.last_input[0].shape)
        # print('mse lastinput[target]: ',self.last_input[1].shape)
        # print('mse gradwrtinput: ', gradwrtinput.shape)
        # print('--------------------------------------------')
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class Sequential(Module):

    def __init__(self, *modules):
        '''
        worch.nn.Sequential class: Pytorch-like Sequential module to stack multiple modules
        (inherited from worch.nn.Module)

        Parameters:
        -----------
        *modules: worch.nn.Module
            Iterable of modules to stack

        Attributes:
        -----------
        module_list: list
            List of modules inside the sequential module.
        '''
        super().__init__()
        assert isinstance(modules[0], Module), '`modules` must contain only Module objects'
        self.module_list = list(modules)
        self.params = self.module_list[0].params
        # Fill previous_module fields to connect them for backprop
        for i in range(1, len(self.module_list)):
            previous_module = self.module_list[i-1]
            module = self.module_list[i]
            assert isinstance(module, Module), '`modules` must contain only Module objects'
            module.register_previous_module(previous_module)
            self.params.extend(module.params)
        
    def __getitem__(self, idx):
        return self.module_list[idx]

    def forward(self, *input):
        '''
        worch.nn.Sequential.forward method: forward pass. Sequentially forwarded through all modules.

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        x: torch.Tensor
            Final output
        '''
        x, = input
        for module in self.module_list:
            # Call each module sequentially
            x = module(x)
        return x
    
    def backward(self, gradwrtoutput):
        '''
        worch.nn.Sequential.backward method: backward pass.

        Parameters:
        -----------
        *input: tuple
            Tensor to input to the module

        Returns:
        --------
        last_module.backward(gradwrtoutput)
            Propagate through modules starting at the last in the stack
        '''
        last_module = self.module_list[-1]
        return last_module.backward(gradwrtoutput)

    def eval(self):
        '''
        worch.nn.Sequential.eval method: set each module to eval mode

        Parameters:
        -----------
        [No parameter]
        '''
        for module in self.module_list:
            module.eval()
        
    def train(self):
        '''
        worch.nn.Sequential.train method: set each module to training mode

        Parameters:
        -----------
        [No parameter]
        '''
        for module in self.module_list:
            module.train()
