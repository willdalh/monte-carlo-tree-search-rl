import numpy as np
import torch
import torch.nn.functional as F
from .node import Node
from statemanagers.state_manager import StateManager

class MonteCarloTree:
    def __init__(self, action_space):
        self.root = None
        self.action_space = action_space
    
    def set_root(self, state):
        if isinstance(state, Node):
            self.root = state
            self.root.parent = None
            # self.root.origin_action = None
        else:
            self.root = Node(parent=None, origin_action=None, state=state)

    def expand_to_depth(self, sm: StateManager, depth):
        self._expand(sm, depth, self.root)
    
    def _expand(self, sm: StateManager, depth, parent):
        # Recursively expand nodes to a certain depth
        # print(parent.state)
        # print(parent.origin_action)
        if depth == 0 or sm.is_final(parent.state):
            return
        # print(sm.is_final(parent.state))
        # TODO perform check whether parent is final. Then no children exists. EDIT: Happens automatically
        if len(parent.children) == 0: # Refer to statemanager to generate children states
            children_states, moves = sm.get_successor_states(parent.state, return_moves=True)
            for state, move in zip(children_states, moves):
                parent.add_child(Node(parent=parent, origin_action=move, state=state))
        
        for child in parent.children:
            self._expand(sm, depth - 1, child)


    def tree_search(self):
        # Find a leaf node and return it so the ANET can perform rollout on it 
        curr_node = self.root
        while len(curr_node.children) != 0:
            Qs = curr_node.get_Qs()
            us = curr_node.get_us()
            curr_player = curr_node.state[0]
            
            child_selected_index = None
            if curr_player == 1:
                child_selected_index = np.argmax([q + u for q, u in zip(Qs, us)])
            elif curr_player == 2:
                child_selected_index = np.argmin([q - u for q, u in zip(Qs, us)])
            curr_node = curr_node.children[child_selected_index]
        return curr_node


    def backpropagate(self, leaf, Z):
        curr_node = leaf
        while curr_node.parent != None: # Propagate upwards until root is reached
            parent = curr_node.parent
            child_index = parent.children.index(curr_node)
            curr_node.N += 1
            parent.Ns[child_index] += 1
            parent.Es[child_index] += Z
            curr_node = parent
        
    def get_visit_distribution(self):
        visit_counts = self.root.Ns
        # print('\n')
        # print('Current state', self.root.state)
        # print('Visits:', visit_counts)
        dist = F.softmax(torch.Tensor(visit_counts), dim=0)
        moves = [child.origin_action for child in self.root.children] # Serves as indices for the final list
        # print('Moves', moves)
        # print('Child states:', [child.state for child in self.root.children])
        final_dist = np.zeros(len(self.action_space))
        for i, move in enumerate(moves):
            final_dist[move] = dist[i]

        return final_dist
        
