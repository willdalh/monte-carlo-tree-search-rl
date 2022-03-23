import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.cases_added = 0

    def store_case(self, case):
        if self.cases_added < self.max_size:
            self.buffer.append(case)
        else:
            self.buffer[self.cases_added%self.max_size] = case
        self.cases_added += 1

    def sample(self, batch_size):
        # TODO: Rework to make faster
        if batch_size > self.cases_added:
            batch_size = self.cases_added
        indices = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]  

        # Create tensors
        states = torch.Tensor([state for state, dist in batch])
        targets = torch.Tensor(np.array([dist for state, dist in batch]))      
        return states, targets

    def get_histogram(self):
        states = [state for state, D in self.buffer]
        uniques = np.unique(states, axis=0)
        counts = {}
        for u in uniques:
            counts[tuple(u.tolist())] = 0

        for state in states:
            counts[tuple(state)] += 1

        res = ''
        for key, val in counts.items():
            res += f'{list(key)}: {val}\n'
        return res

        
class ReplayBufferTensor:
    def __init__(self, max_size, state_size, action_space_size):
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.states = torch.zeros(0, state_size)
        self.targets = torch.zeros(0, action_space_size)
        self.expansion_size = 200
        self.max_size = max_size
        self.cases_added = 0


    def store_case(self, case):
        if self.cases_added >= self.states.shape[0]:
            self._expand_buffers()
        state, target = case
        for i in range(len(state)):
            self.states[self.cases_added%self.max_size, i] = state[i]
        for i in range(len(target)):
            self.targets[self.cases_added%self.max_size, i] = target[i]
        self.cases_added += 1

    def _expand_buffers(self):
        if self.states.shape[0] < self.max_size:
            self.states = torch.cat([self.states, torch.zeros(self.expansion_size, self.state_size)], dim=0)
            self.targets = torch.cat([self.targets, torch.zeros(self.expansion_size, self.action_space_size)], dim=0)

    def sample(self, batch_size):
        if batch_size > self.cases_added:
            batch_size = self.cases_added
        num_stored_cases = self.cases_added if self.cases_added < self.max_size else self.max_size
        indices = np.random.choice(range(num_stored_cases), size=batch_size, replace=False)

        states = self.states[indices]
        targets = self.targets[indices] 

        return states, targets

    def get_histogram(self):
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
    