import torch
torch.set_grad_enabled(False)

class Module(object):

    '''
    Parent class to inherit from
    '''

    def __init__(self):
        self.last_input = None
        self.previous_module = None
        self.params = []
        self.training = True
        self.device = torch.device('cpu')

    def forward(self, *input):
        raise NotImplementedError('No implementation of forward found')

    def __call__(self, *input):
        return self.forward(*input)
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError('No implementation of backward found')

    def initialize_parameters(self):
        raise NotImplementedError('No implementation of initialize_parameters found')

    def to(self, device):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        for i in range(len(self.params)):
            if len(self.params[i]) > 0:
                self.params[i] = self.params[i].to(device)
        return self

    def parameters(self):
        return self.params

    def register_previous_module(self, module):
        assert isinstance(module, Module), f'module must be Module.'
        self.previous_module = module

    def propagate_gradient(self, gradwrtinput):
        if self.previous_module is None:
            return gradwrtinput
        else:
            return self.previous_module.backward(gradwrtinput)

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        weights = torch.zeros((self.out_features, self.in_features))
        if self.use_bias:
            bias = torch.zeros((self.out_features, 1))
            self.params = [weights, bias]
            
        else:
            self.params = [weights]
        self.initialize_parameters()

    def forward(self, *input):
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
        gradwrtoutput, = gradwrtoutput
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
        self.params[0] = torch.nn.init.xavier_uniform_(self.params[0])
        self.params[1] = torch.nn.init.xavier_uniform_(self.params[1]).squeeze(-1)

class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        return torch.clamp(x, min=0)

    def backward(self, *gradwrtoutput):
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
        super().__init__()

    def forward(self, *input):
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
        gradwrtoutput, = gradwrtoutput
        output_sigmoid = self.forward(self.last_input)
        gradwrtinput = (output_sigmoid*(1-output_sigmoid))*gradwrtoutput
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class Tanh(Module):

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        if self.training:
            self.last_input, = input
            x = self.last_input.clone()
        else:
            x, = input
        return (torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x))

    def backward(self, *gradwrtoutput):
        gradwrtoutput, = gradwrtoutput
        output_tanh = self.forward(self.last_input)
        gradwrtinput = (1-torch.pow(output_tanh, 2))*gradwrtoutput
        self.last_input = None
        return self.propagate_gradient(gradwrtinput)

class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        target = target.reshape(input.shape) # just in case
        if self.training:
            self.last_input = [input, target]
        agg = torch.pow(input-target, 2)
        return torch.mean(agg)

    def backward(self):
        '''loss function => no gradwrtoutput'''
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
        super().__init__()
        assert isinstance(modules[0], Module), '`modules` must contain only Module objects'
        self.module_list = list(modules)
        self.params = self.module_list[0].params
        for i in range(1, len(self.module_list)):
            previous_module = self.module_list[i-1]
            module = self.module_list[i]
            assert isinstance(module, Module), '`modules` must contain only Module objects'
            module.register_previous_module(previous_module)
            self.params.extend(module.params)
        
    def __getitem__(self, idx):
        return self.module_list[idx]

    def forward(self, *input):
        x, = input
        for module in self.module_list:
            x = module(x)
        return x
    
    def backward(self, gradwrtoutput):
        last_module = self.module_list[-1]
        return last_module.backward(gradwrtoutput)

    def eval(self):
        for module in self.module_list:
            module.eval()
        
    def train(self):
        for module in self.module_list:
            module.train()
