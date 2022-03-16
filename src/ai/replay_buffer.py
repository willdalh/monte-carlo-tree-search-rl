import numpy as np

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
        return batch

    def print_histogram(self):
        states = [state for state, D in self.buffer]
        uniques = np.unique(states, axis=0)
        counts = {}
        for u in uniques:
            counts[tuple(u.tolist())] = 0

        for state in states:
            counts[tuple(state)] += 1

        for key, val in counts.items():
            print(f'{list(key)}: {val}')

        



    