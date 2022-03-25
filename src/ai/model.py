import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from collections import OrderedDict
import numpy as np
import logging

class Model(nn.Module):
    '''Neural network'''
    def __init__(self, nn_dim, optimizer_name, lr):
        super(Model, self).__init__()
        self.nn_dim = nn_dim
        self.lr = lr
        has_conv = np.any(['conv' in str(e) for e in nn_dim])

        if has_conv:
            layers = Model.generate_layers_conv(nn_dim)
        else:
            if not isinstance(nn_dim[1], int) :
                raise ValueError('First element specified in nn_dim must be an integer')
            layers = Model.generate_layers(nn_dim)
        logging.debug(nn_dim)
        logging.debug('\n')
        logging.debug('Constructed the following layers:')
        [logging.debug(e[-1]) for e in layers]
        logging.debug('\n')

        od = OrderedDict(layers)
        self.net = nn.Sequential(od)
        
        self.optimizer = None
        self.initialize_optimizer(optimizer_name)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss()
        

    def forward(self, x):
        return F.softmax(self.logits(x), dim=1)

    def logits(self, x):
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
        param_count = 0
        for name, layer in list(self.net.named_parameters())[::2]:
            string += f'{name} with shape {layer.shape}\n'
        
        for name, layer in list(self.net.named_parameters()):
            param_count += torch.flatten(layer.data).shape[0]

        
        return string + f'Number of parameters: {param_count}'

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
        structure = [e for e in nn_dim if isinstance(e, int)] # Get hidden layer sizes by filtering out activation functions
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
        # layers.append(('softmax_end', nn.Softmax(dim=1)))
        # layers_test.append('softmax')
        logging.debug('nn_dim interpreted as:')
        logging.debug(layers_test)
        print('nn_dim interpreted as:', layers_test, '\n')

        return layers


    def generate_layers_conv(nn_dim):
        '''
        Generate a list containing layers and activation functions in the order specified in the input
        
        Args:
            nn_dim: List or tuple of layer sizes and activations functions (e.g. [16, 32, 'sigmoid', 4, 'softmax'])
        
        Returns:
            List of tuples, where each tuple contains the name and an instantiated object of either a linear layer or an activation function
        '''
        layers = []
        layers_test = []

        # Create conv-layers
        height_width = int(np.sqrt(int(nn_dim[0]))) # Height and width of the quadratic-sized feature map
        # layers.append(('unflatten_start', nn.Unflatten(1, (1, height_width, height_width))))
        
        nn_dim[0] = 'unflatten'
        last_conv_index = 0
        for i, e in enumerate(nn_dim):
            if 'conv' in str(e):
                last_conv_index = i
        nn_dim.insert(last_conv_index+1, 'flatten')

        print(nn_dim)

        # layers_test.append(f'unflatten(1-{height_width}-{height_width})')
        for i, elem in enumerate(nn_dim):
            if elem == 'unflatten':
                layers.append(('unflatten_start', nn.Unflatten(1, (1, height_width, height_width))))
                layers_test.append(f'unflatten(1-{height_width}-{height_width})')

            if 'conv' in str(elem):
                config = elem[5:-1].split('-')
                conv_args = {'k': 3, 'p': 1, 's': 1} # Default
                for e in config:
                    conv_args[e[0]] = int(e[1:])
                height_width = ((height_width - conv_args['k'] + 2 * conv_args['p'])/conv_args['s']) + 1

                # Find in-channels
                in_channels = 1
                if len(layers) > 1:
                    in_channels = layers[-1][-1].out_channels
                layers.append((f'conv{i}', nn.Conv2d(in_channels=in_channels, out_channels=conv_args['c'], kernel_size=conv_args['k'], stride=conv_args['s'], padding=conv_args['p'])))
                layers_test.append(elem)

            if  elem == 'flatten':
                flattened_size = int(height_width ** 2) * layers[-1][-1].out_channels
                layers.append((f'flatten_end', nn.Flatten(1, 3)))
                layers_test.append(f'flatten({flattened_size})')
            

        # print(layers_test)
        # quit()

        # flattened_size = int(height_width ** 2) * layers[-1][-1].out_channels
        # layers.append((f'flatten_end', nn.Flatten(1, 3)))
        # layers_test.append(f'flatten({flattened_size})')
        # W_out = ((W_in - F + 2*P)/S) + 1
        
    
        structure = [e for e in nn_dim if isinstance(e, int)] # Get hidden layer sizes by filtering out activation functions
        structure.insert(0, flattened_size)
        print(structure)
        # Initialize linear layers and store them in the list layers
        for i, (in_neurons, out_neurons) in enumerate(zip(structure, structure[1:])):
            layers.append((f'layer{i+1}', nn.Linear(in_neurons, out_neurons)))
            layers_test.append((in_neurons, out_neurons))
        
        # Find activation functions
        for i, e in enumerate(nn_dim):
            if isinstance(e, str) and 'conv' not in str(e) and 'flatten' not in str(e): # Handle activation functions
                act_func = e.lower()
                # Initialize activation functions and store them at the correct place in the list
                layers.insert(i, (f'{act_func}{i-1}', Model.get_activation_function(act_func)))
                layers_test.insert(i, e)
        logging.debug('nn_dim interpreted as:')
        logging.debug(layers_test)
        # print('nn_dim interpreted as:', layers_test, '\n')
        # print([e[-1] for e in layers])

        return layers

if __name__ == '__main__':
    nn_dim = '9,conv(k3-p1-s1-c5),relu,conv(k3-p2-s1-c3),256,relu,9'
    nn_dim = '9,conv(k3-p1-s1-c5),relu,conv(k3-p3-s1-c3),relu,256,relu,512,relu,9'
    nn_dim = '9,conv(c1),16,relu,32,9,relu'
    nn_dim = '9,conv(c5),relu,400,relu,256,relu'
    # Result: unflatten(sqrt(9), sqrt(9)), conv(k3,p1,s1,c5), flatten(), lin(9, 256), relu, lin(256, 9)

    # W_out = ((W_in - F + 2*P)/S) + 1

    nn_dim = [int(e) if e.isdigit() else e for e in nn_dim.split(',')]
    # nn_dim = [int(e) for e in ','.split(nn_dim) if e.isdigit()]
    # print([e[-1] for e in Model.generate_layers_conv(nn_dim)])
    model = Model(nn_dim, 'adam', 0.001)
    tensor_test = torch.rand(4, 9)
    print(model(tensor_test))