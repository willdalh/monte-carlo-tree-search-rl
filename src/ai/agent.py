import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from statemanagers.state_manager import StateManager

from .model import Model
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, buffer_size, batch_size, lr, nn_dim, optimizer, epsilon_decay, **_):
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.anet = Model(nn_dim=nn_dim, optimizer_name=optimizer, lr=lr)

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.mse_losses = []
        self.cross_losses = []

    def train_on_buffer_batch(self, sm: StateManager, debug=False):
        minibatch = self.buffer.sample(self.batch_size)
        states = torch.Tensor([state for state, dist in minibatch]) #TODO: legal moves fix
        target = torch.Tensor(np.array([dist for state, dist in minibatch]))
        prediction = self.anet(states)
        if debug:
            print('\n')
            print(f'state: {states[0]}')
            print(f'pred: {prediction[0]}')
            print(f'tar: {target[0]}')
        loss = self.anet.loss_fn(prediction, target)
        loss.backward()
        self.anet.optimizer.step()
        self.anet.optimizer.zero_grad()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        
        self.cross_losses.append(loss.item())
        with torch.no_grad():
            self.mse_losses.append(F.mse_loss(prediction, target).item())

    def store_case(self, case):
        self.buffer.store_case(case)

    def choose_action(self, state, legal_moves=None, debug=False):
        action = None
        if np.random.random() > self.epsilon: # exploit
            dist = self.anet.forward(torch.Tensor([state]))[0]
            for i in range(dist.shape[0]):
                if i not in legal_moves:
                    dist[i] = 0
            dist *= 1/dist.sum()
            if debug:
                print('dist:', dist)
            action = torch.argmax(dist).item()
        else: # explore
            action = np.random.choice(legal_moves)

        return action

    def rollout(self, sm: StateManager, leaf):
        state = leaf.state
        while not sm.is_final(state): # Rollout to final state
            legal_moves = sm.get_legal_moves(state)
            action = self.choose_action(state, legal_moves) 
            state = sm.get_successor(state, action)
        winner = sm.get_winner(state)
        Z = 1 if winner == 1 else -1 # Evaluation of final state
        return Z
    
    def load_model(self, path):
        self.anet.load_model(path)

    def present_results(self):
        print('\nLength of buffer:', len(self.buffer.buffer))
        plt.plot(np.arange(len(self.mse_losses)), np.array(self.mse_losses))
        # plt.plot(np.arange(len(self.cross_losses)), np.array(self.cross_losses))
        plt.show()


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
