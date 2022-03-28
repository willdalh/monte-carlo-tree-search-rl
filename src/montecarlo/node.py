
import numpy as np

class Node:
    def __init__(self, parent, origin_action, state, c):
        self.parent = parent
        self.origin_action = origin_action # The action that led to this node
        self.state = state #[pid, *board]
        
        self.N = 0 # Count for number of visits

        self.children = []
        self.Ns = [] # Count for number of visits to each child
        self.Es = [] # Sum of evaluations for each child

        self.c = c

    def add_child(self, child):
        '''Add a child to this node'''
        self.children.append(child)
        self.Ns.append(0)
        self.Es.append(0)

    def get_Qs(self):
        '''Calculate the Q values for each child'''
        return [e / n if n != 0 else 0 for e, n in zip(self.Es, self.Ns)]

    def get_us(self):
        '''Calculate the exploration values for each child'''
        if self.N == 0: # Avoid square root of -inf
            return [0 for i in range(len(self.children))]
            
        res = [self.c * np.sqrt((np.log(self.N))/(1 + self.Ns[i])) for i in range(len(self.Ns))]
        return res
        

