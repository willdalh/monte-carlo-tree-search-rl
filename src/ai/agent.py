import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging

from statemanagers.state_manager import StateManager
from statemanagers.hex_state_manager import HEXStateManager

from .model import Model
from .replay_buffer import ReplayBuffer, ReplayBufferTensor

import time


class Agent:
    def __init__(self, buffer_size, batch_size, lr, nn_dim, optimizer, epsilon_decay, state_size, action_space_size, **_):
        self.batch_size = batch_size
        # self.buffer = ReplayBuffer(buffer_size)
        self.buffer = ReplayBufferTensor(buffer_size, state_size, action_space_size)
        self.anet = Model(nn_dim=nn_dim, optimizer_name=optimizer, lr=lr)
        logging.debug('Anet initialized to:')
        logging.debug(self.anet.to_string())

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.mse_losses = []
        self.cross_losses = []
        self.exploits_done = 0

        self.demonstrate_epsilon()
        

    def train_on_buffer_batch(self, epochs=10, debug=False):
        losses = []
        # if self.buffer.cases_added < 30: return -1
        for epoch in range(epochs):
            
            states, targets = self.buffer.sample(self.batch_size)
            # targets = torch.argmax(targets, dim=1)
            prediction = self.anet.logits(states)

            loss = self.anet.loss_fn(prediction, targets)
            loss.backward()
            self.anet.optimizer.step()
            self.anet.optimizer.zero_grad()

            if debug and epoch > 6:
                with torch.no_grad():
                    logging.debug('')
                    logging.debug(f'state: {states[0]}')
                    logging.debug(f'pred dist: {F.softmax(prediction[0], dim=0)}')
                    logging.debug(f'tar: {targets[0]}')
        
            with torch.no_grad():
                losses.append(loss.item())
                # self.mse_losses.append(F.mse_loss(prediction, target).item())
        mean_loss = np.mean(losses)
        self.cross_losses.append(mean_loss)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        return mean_loss

    def store_case(self, case, sm: StateManager):
        self.buffer.store_case(case)
        
        if isinstance(sm, HEXStateManager):
            state, D = case
            symmetric_state = state[-sm.K * sm.K:][::-1] # Rotate 180 degrees
            symmetric_state.insert(0, state[0]) # Insert current player
            symmetric_D = D[::-1] # Rotate 180 degrees
            self.buffer.store_case((symmetric_state, symmetric_D))
            

    def choose_action(self, state, legal_moves, debug=False):
        action = None
        if np.random.random() > self.epsilon: # exploit
            dist = self.anet.forward(torch.Tensor([state]))[0]
            for i in range(dist.shape[0]):
                if i not in legal_moves:
                    dist[i] = 0.0
            dist *= 1/dist.sum()
            if debug:
                logging.debug(f'dist: {dist}')
            action = torch.argmax(dist).item()
            self.exploits_done += 1
        else: # explore
            action = np.random.choice(legal_moves)

        return action

    
    def load_model(self, path):
        self.anet.load_model(path)

    def demonstrate_epsilon(self):
        eps = self.epsilon
        decay = self.epsilon_decay
        count = 0
        while eps > self.min_epsilon:
            eps *= decay
            count += 1
            if count%5 == 0:
                print(f'count {count}: eps = {eps}')
        print(f'count {count}: eps = {eps}')


    def get_accuracy_on_buffer(self):
        states, targets = self.buffer.sample(self.buffer.cases_added)
        targets = torch.argmax(targets, dim=1)
        pred = self.anet.forward(states)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == targets).sum() / (pred == targets).shape[0]
        return acc.item()


    def present_results(self, log_dir):
        logging.debug(f'Exploits done: {self.exploits_done}')
        logging.debug(f'Replay histogram:\n{self.buffer.get_histogram()}')
        logging.debug(f'Accuracy on replay buffer is: {self.get_accuracy_on_buffer()}')
        
        # print('\nLength of buffer:', len(self.buffer.buffer))
        # plt.plot(np.arange(len(self.mse_losses)), np.array(self.mse_losses))
        plt.plot(np.arange(len(self.cross_losses)), np.array(self.cross_losses))
        plt.title('Cross-entropy loss')
        plt.savefig(f'{log_dir}/cross_loss.png', dpi=300)
        plt.show()


class PreTrainedAgent:
    def __init__(self, model_path, nn_dim):
        self.anet = Model(nn_dim=nn_dim, optimizer_name='adam', lr=0.0000001)
        self.anet.load_model(model_path)

        self.index = -1

    def choose_action(self, state, legal_moves):
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
