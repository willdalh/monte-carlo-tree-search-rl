import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging

from statemanagers.state_manager import StateManager
from statemanagers.hex_state_manager import HEXStateManager

from .model import Model
from .replay_buffer import ReplayBufferTensor



class Agent:
    def __init__(self, buffer_size, batch_size, lr, nn_dim, optimizer, epsilon, epsilon_decay, pre_trained_path, state_size, action_space_size, **_):
        '''
        Initialize an agent.
        The agent will always play from the perspective of player 1.

        Args:
            buffer_size: The size of the replay buffer.
            batch_size: The size of the batch to be used for training.
            lr: The learning rate for the ANET.
            nn_dim: The structure of the ANET.
            optimizer: A string indicating the optimizer to be used.
            epsilon: The initial epsilon value.
            epsilon_decay: The decay rate for epsilon every episode.
            pre_trained_path: Path to a pre-trained model. If given, this model will be loaded.
            state_size: The size of the state when excluding the player id.
            action_space_size: The size of the action space.
            _: Unused keyword arguments.
        '''

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.buffer = ReplayBufferTensor(buffer_size, state_size, action_space_size)
        self.anet = Model(nn_dim=nn_dim, optimizer_name=optimizer, lr=lr)
        if pre_trained_path != None:
            self.anet.load_model(pre_trained_path)
        logging.debug('Anet initialized to:')
        logging.debug(self.anet.to_string())

        self.mse_losses = []
        self.cross_losses = []
        self.exploits_done = 0
        self.min_epsilon = 0.01

        self.demonstrate_epsilon()
        

    def train_on_buffer_batch(self, epochs=10, debug=False):
        '''
        Retrieve a batch of cases from the replay buffer and train the ANET on them.'''
        losses = []
        for epoch in range(epochs):
            states, targets = self.buffer.sample(self.batch_size)
            # targets = torch.argmax(targets, dim=1)
            prediction = self.anet.logits(states)

            loss = self.anet.loss_fn(prediction, targets)
            loss.backward()
            self.anet.optimizer.step()
            self.anet.optimizer.zero_grad()
        
            with torch.no_grad():
                losses.append(loss.item())
 
        mean_loss = np.mean(losses)
        self.cross_losses.append(mean_loss)

        # with torch.no_grad():
        #     logging.debug('')
        #     logging.debug(f'state: {states[0]}')
        #     logging.debug(f'pred dist: {F.softmax(prediction[0], dim=0)}')
        #     logging.debug(f'tar: {targets[0]}')
        
        return mean_loss

    def decay_epsilon(self):
        '''Decay the epsilon value.'''
        self.epsilon *= self.epsilon_decay

    def store_case(self, case):
        '''
        Store a case in the replay buffer.

        Args:
            case: tuple of (state, distribution) where the state is from the perspecitve of player 1.
        '''
        self.buffer.store_case(case)


    def choose_action(self, state, legal_moves, debug=False):
        '''
        Choose an action to play from the perspective of player 1 using an epsilon-greedy policy.

        Args:
            state: The state of the game from the perspective of player 1. 
            legal_moves: The legal moves from the current state.
        
        Returns:
            The action to be played from the perspective of player 1.
        '''
        action = None
        if np.random.random() > self.epsilon: # Exploit
            dist = self.anet.forward(torch.Tensor([state]))[0]
            for i in range(dist.shape[0]): # Set the probability of illegal moves to 0
                if i not in legal_moves:
                    dist[i] = 0.0
            dist *= 1/dist.sum() # Rescale to sum to 1
            if debug:
                logging.debug(f'dist: {dist}')
            action = torch.argmax(dist).item() # Choose the action with the highest probability
            self.exploits_done += 1
        else: # Explore
            action = np.random.choice(legal_moves)

        return action

    
    def load_model(self, path):
        '''Load a model from a file.'''
        self.anet.load_model(path)

    def demonstrate_epsilon(self):
        '''For debugging purposes. Print how the epsilon value evolves.'''
        eps = self.epsilon
        decay = self.epsilon_decay
        count = 0
        epsilon_series = []
        while eps > self.min_epsilon:
            eps *= decay
            epsilon_series.append((count, eps))
            count += 1

        length = len(epsilon_series)
        for i, (count, eps) in enumerate(epsilon_series):
            if i%(length//15) == 0:
                print(f'count {count}: eps = {eps}')


    def get_accuracy_on_buffer(self):
        '''Calculate the accuracy of the ANET on the replay buffer.'''
        states, targets = self.buffer.sample(self.buffer.cases_added)
        targets = torch.argmax(targets, dim=1)
        pred = self.anet.forward(states)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == targets).sum() / (pred == targets).shape[0]
        return acc.item()


    def present_results(self, log_dir, display):
        '''Present the results of the training session.'''
        torch.save(self.buffer.states, f'{log_dir}/buffer_states.pt')
        torch.save(self.buffer.targets, f'{log_dir}/buffer_targets.pt')
        logging.debug(f'Exploits done: {self.exploits_done}')
        logging.debug(f'Replay histogram:\n{self.buffer.get_histogram()}')
        logging.debug(f'Accuracy on replay buffer is: {self.get_accuracy_on_buffer()}')
        
        # print('\nLength of buffer:', len(self.buffer.buffer))
        plt.plot(np.arange(len(self.cross_losses)), np.array(self.cross_losses))
        plt.title('Cross-entropy loss on replay buffer samples')
        plt.savefig(f'{log_dir}/cross_loss.png', dpi=300)
        if display:
            plt.show()



class PreTrainedAgent:
    def __init__(self, model_path, nn_dim):
        '''
        Initialize a pre-trained agent.
        
        Args:
            model_path: Path to a saved model.
            nn_dim: The structure of the ANET.
        '''
        self.anet = Model(nn_dim=nn_dim, optimizer_name='adam', lr=0.0000001)
        self.anet.load_model(model_path)

        self.index = -1

    def choose_action(self, state, legal_moves):
        '''
        Choose an action to play from the perspective of player 1. Same as in the Agent class, but this is only greedy.

        Args:
            state: The state of the game from the perspective of player 1.
            legal_moves: The legal moves from the current state.
        
        Returns:
            The action to be played from the perspective of player 1.
        '''
        dist = self.anet.forward(torch.Tensor([state]))[0]
        for i in range(dist.shape[0]):
            if i not in legal_moves:
                dist[i] = 0.0
        dist *= 1/dist.sum()
        action = torch.argmax(dist).item()
        return action
        


#%%

# import torch
# import numpy as np

# state_np = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# state = list(range(10))

# print('Time for creating tensor from numpy')
# %timeit -n 10 [torch.Tensor([state_np]) for i in range(100)]

# print('Time for creating tensor from python list')
# %timeit -n 10 [torch.Tensor([state]) for i in range(100)]


# %%
# import torch
# import numpy as np

# print('Time for creating many lists')
# %timeit -n 20 [[1, 2, 3, 4] for i in range(10000)]

# print('Time for creating many np arrays')
# %timeit -n 20 [np.array([1, 2, 3, 4]) for i in range(10000)]

# print('Time for creating many tensors')
# %timeit -n 20 [torch.Tensor([1, 2, 3, 4]) for i in range(10000)]

# %%
# data = [i for i in range(10000)]
# data_set = set(data)
# print('Time for iterating list:')
# %timeit -n 100 [i for i in data]

# print('Time for iterating set')
# %timeit -n 100 [i for i in data_set]

# %%
# %timeit -n 100 [i**2 for i in range(100000)]



# %%
# import numpy as np
# state = [0, 1, 1, -1, 0, -1, 0, 1, -1]
# K = 3
# print('Time for flipping with Python')
# %timeit -n 1000 [-v for i in range(K) for v in state[i::K]]

# print('Time for flipping with Numpy')
# %timeit -n 1000 (np.array(state).reshape(K, K).T.ravel() * - 1).tolist()



# %%
