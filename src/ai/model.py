import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from collections import OrderedDict

class Model(nn.Module):
    '''Neural network'''
    def __init__(self, nn_dim, optimizer_name, lr):
        super(Model, self).__init__()
        self.nn_dim = nn_dim
        self.lr = lr

        if not isinstance(nn_dim[1], int) :
            raise ValueError('First element specified in nn_dim must be an integer')

        layers = Model.generate_layers(nn_dim)

        od = OrderedDict(layers)
        self.net = nn.Sequential(od)
        
        self.optimizer = None
        self.initialize_optimizer(optimizer_name)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss()
        

    def forward(self, x):
        return self.net(x.float())

    def initialize_optimizer(self, optimizer_name):
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Specified optimizer \'{optimizer_name}\' is not supported')

    def to_string(self):
        '''Present the neural network structure, without the bias terms'''
        string = ''
        for name, layer in list(self.net.named_parameters())[::2]:
            string += f'{name} with shape {layer.shape}\n'
        return string

    def save_model(self, path, name):
        '''Save the model parameters to file'''
        torch.save(self.state_dict(), f'{path}/{name}')

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


    @staticmethod
    def get_activation_function(func_str):
        '''
        Return an instance of an activation function given the name
        
        Args: 
            func_str: name of the activation function

        Returns: 
            An instance of an activation function
        '''
        if func_str == 'sigmoid':
            return nn.Sigmoid()
        elif func_str == 'tanh':
            return nn.Tanh()
        elif func_str == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f'Specified activation function \'{func_str}\' is not supported')

    @staticmethod
    def generate_layers(nn_dim):
        '''
        Generate a list containing layers and activation functions in the order specified in the input
        
        Args:
            nn_dim: List or tuple of layer sizes and activations functions (e.g. [16, 32, 'sigmoid', 4, 'softmax'])
        
        Returns:
            List of tuples, where each tuple contains the name and an instantiated object of either a linear layer or an activation function
        '''
        structure = [e for e in nn_dim if isinstance(e, int)] # Get hidden layer sizes, aka filter out activation functions
        layers = []
        layers_test = []

        # Initialize linear layers and store them in the list layers
        for i, (in_neurons, out_neurons) in enumerate(zip(structure, structure[1:])):
            layers.append((f'layer{i+1}', nn.Linear(in_neurons, out_neurons)))
            layers_test.append((in_neurons, out_neurons))
        
        # Iterate specified neural network structure
        for i, e in enumerate(nn_dim):
            if isinstance(e, str): # Handle activation functions
                act_func = e.lower()
                # Initialize activation functions and store them at the correct place in the list
                layers.insert(i-1, (f'{act_func}{i-1}', Model.get_activation_function(act_func)))
                layers_test.insert(i-1, e)
        layers.append(('softmax_end', nn.Softmax(dim=1)))
        layers_test.append('softmax')
        print('nn_dim interpreted as:', layers_test, '\n')

        return layers

# class PreTrainedModel:
#     def __init__(self, nn_dim):
#         layers = Model.generate_layers(nn_dim)

