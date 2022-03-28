import numpy as np
import torch
        
class ReplayBufferTensor:
    def __init__(self, max_size, state_size, action_space_size):
        '''
        Initialize the replay buffer

        Args:
            max_size: the maximum size of the buffer
            state_size: the size of the flattened state excluding the player id
            action_space_size: the size of the action space
        '''
        self.max_size = max_size
        self.state_size = state_size
        self.action_space_size = action_space_size 

        self.states = torch.zeros(0, state_size) # Buffer for states
        self.targets = torch.zeros(0, action_space_size) # Buffer for targets

        self.expansion_size = 200
        self.cases_added = 0

    def store_case(self, case):
        '''
        Add a case to the buffer
        
        Args:
            case: a tuple of (flattened board state, target distribution) 
        '''
        if self.cases_added >= self.states.shape[0]: # Expand tensor buffers if necessary
            self._expand_buffers()
        state, target = case
        for i in range(len(state)):
            self.states[self.cases_added%self.max_size, i] = state[i]
        for i in range(len(target)):
            self.targets[self.cases_added%self.max_size, i] = target[i]
        self.cases_added += 1

    def _expand_buffers(self):
        '''Concatenate empty tensors to the end of the buffers'''
        if self.states.shape[0] < self.max_size:
            self.states = torch.cat([self.states, torch.zeros(self.expansion_size, self.state_size)], dim=0)
            self.targets = torch.cat([self.targets, torch.zeros(self.expansion_size, self.action_space_size)], dim=0)

    def sample(self, batch_size):
        '''
        Sample a batch of random cases from the buffer
        
        Args:
            batch_size: the amount of cases to sample

        Returns:
            states: tensor of shape (batch_size, state_size)
            targets: tensor of shape (batch_size, action_space_size)
        '''
        if batch_size > self.cases_added:
            batch_size = self.cases_added
        num_stored_cases = self.cases_added if self.cases_added < self.max_size else self.max_size
        indices = np.random.choice(range(num_stored_cases), size=batch_size, replace=False)

        states = self.states[indices]
        targets = self.targets[indices] 

        return states, targets

    def get_histogram(self):
        '''Return a histogram of the buffer'''
        uniques = torch.unique(self.states, dim=0)
        counts = {}
        for u in uniques:
            if not torch.all(u == torch.zeros(u.shape[0])):
                counts[tuple(map(int, u.tolist()))] = torch.count_nonzero(torch.all(self.states == u, dim=1))

        res = ''
        counts_sorted = dict(sorted(counts.items(), key=lambda x: x[1]))
        for key, val in counts_sorted.items():
            res += f'{list(key)}: {val}\n'
        return res
    