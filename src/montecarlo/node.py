
import numpy as np

class Node:
    def __init__(self, parent, origin_action, state, c):
        self.parent = parent
        self.origin_action = origin_action
        self.state = state
        

        self.N = 1

        self.children = []
        self.Ns = []
        self.Es = []

        self.c = c

    def add_child(self, child):
        self.children.append(child)
        self.Ns.append(1)
        self.Es.append(0)

    def get_Qs(self):
        return [e / n for e, n in zip(self.Es, self.Ns)]

    def get_us(self):
        return [self.c * np.sqrt((np.log(self.N))/(1 + self.Ns[i])) for i in range(len(self.Ns))]
        

