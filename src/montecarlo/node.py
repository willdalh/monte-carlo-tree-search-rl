
import numpy as np

class Node:
    def __init__(self, parent, origin_action, state, c):
        self.parent = parent
        self.origin_action = origin_action
        self.state = state
        

        self.N = 0

        self.children = []
        self.Ns = []
        self.Es = []

        self.c = c

    def add_child(self, child):
        self.children.append(child)
        self.Ns.append(0)
        self.Es.append(0)

    def get_Qs(self):
        return [e / n if n != 0 else 0 for e, n in zip(self.Es, self.Ns)]

    def get_us(self):
        if self.N == 0:
            return [0 for i in range(len(self.children))]
            
        res = [self.c * np.sqrt((np.log(self.N))/(1 + self.Ns[i])) for i in range(len(self.Ns))]
        return res
        

