import torch
import numpy as np

from statemanagers.state_manager import StateManager

from .model import Model
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, buffer_size, batch_size, lr, nn_dim, optimizer, epsilon_decay, **_):
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.anet = Model(nn_dim=nn_dim, optimizer_name=optimizer, lr=lr)

        self.epsilon = 0.9
        self.epsilon_decay = epsilon_decay

    def train_on_buffer_batch(self):
        minibatch = self.buffer.sample(self.batch_size)
        states = torch.Tensor([state for state, dist in minibatch])
        target = torch.Tensor(np.array([dist for state, dist in minibatch]))
        prediction = self.anet(states)
        loss = self.anet.loss_fn(prediction, target)
        loss.backward()
        self.anet.optimizer.step()
        self.anet.optimizer.zero_grad()

    def store_case(self, case):
        self.buffer.store_case(case)

    def choose_action(self, state, legal_moves=None):
        action = None
        if np.random.random() > self.epsilon: # exploit
            dist = self.anet(torch.Tensor([state]))
            action = torch.argmax(dist).item()
            if action not in legal_moves:
                action = np.random.choice(legal_moves)
        else: # explore
            action = np.random.choice(legal_moves)

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
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
