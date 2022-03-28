import time
import numpy as np
import torch
import torch.nn.functional as F
from .node import Node
from statemanagers.state_manager import StateManager
import graphviz
import torch.multiprocessing as tmp

class MonteCarloTree:
    def __init__(self, max_depth, c, action_space, **_):
        '''
        Initialize the Monte Carlo Tree
        
        Args:
        
        '''
        self.max_depth = max_depth
        self.action_space = action_space
        self.c = c
        self.expansion_threshold = 0
        self.root = None

        self.vis_counter = 0

    
    def set_root(self, state):
        if isinstance(state, Node):
            self.root = state
            self.root.parent = None
        else:
            self.root = Node(parent=None, origin_action=None, state=state, c=self.c)

    # def expand_to_depth(self, sm: StateManager):
    #     self._expand(sm, self.max_depth, self.root)
    
    # def _expand(self, sm: StateManager, depth, parent):
    #     # Recursively expand nodes to a certain depth
    #     if depth == 0 or sm.is_final(parent.state):
    #         return

    #     # TODO perform check whether parent is final. Then no children exists. EDIT: Think it happens automatically, check
    #     if len(parent.children) == 0: # Refer to statemanager to generate children states
    #         self.expand_node(parent=parent, sm=sm)
        
    #     for child in parent.children:
    #         self._expand(sm, depth - 1, child)

    def expand_node(self, parent: Node, sm: StateManager): # TODO LOOK INTO CHECKING FOR FINAL STATES WHEN GENERATING SUCCESSORS
        children_states, moves = sm.get_successor_states(parent.state, return_moves=True)
        for state, move in zip(children_states, moves):
            parent.add_child(Node(parent=parent, origin_action=move, state=state, c=self.c))
        return len(children_states) > 0

    # def tree_search(self):
    #     # Find a leaf node and return it so the ANET can perform rollout on it 
    #     curr_node = self.root
    #     while len(curr_node.children) != 0:
    #         Qs = curr_node.get_Qs()
    #         us = curr_node.get_us()
    #         curr_player = curr_node.state[0]

    #         child_selected_index = None
    #         if curr_player == 1:
    #             child_selected_index = np.argmax([q + u for q, u in zip(Qs, us)])
    #         elif curr_player == -1:
    #             child_selected_index = np.argmin([q - u for q, u in zip(Qs, us)])
    #         curr_node = curr_node.children[child_selected_index]

    #     return curr_node

    # def perform_search_game(self, sm: StateManager, agent):
    #     leaf = self.tree_search_expand(sm)
    #     processes = []
    #     if self.use_mp:
    #         for i in range(self.cpu_count):
    #             p = tmp.Process(target=self.fake_method, args=(i, q))
    #             processes.append(p)
    #         for p in processes:
    #             p.start()
            
    #         Zs = []
    #         for p in processes:
    #             p.join()
    #             Z = q.get()
    #             Zs.append(Z)
    #             del Z
    #         self.backpropagate(leaf, Zs)

    #     else:
    #         Z = self.rollout(agent, sm, leaf)
    #         self.backpropagate(leaf, Z)

    def fake_method(self, number, q):
        np.random.seed()
        number += 1
        result = np.random.randint(-1, 2)
        if self.use_mp:
            q.put(result)
            return
        return result

    def tree_search_expand(self, sm: StateManager):
        '''Tree traversal as shown by John Levine on YouTube'''
        curr_node = self.root
        while len(curr_node.children) != 0:
            Qs = curr_node.get_Qs()
            us = curr_node.get_us()
            curr_player = curr_node.state[0]

            child_selected_index = None
            if curr_player == 1:
                child_selected_index = np.argmax([q + u for q, u in zip(Qs, us)])
            elif curr_player == -1:
                child_selected_index = np.argmin([q - u for q, u in zip(Qs, us)])
            curr_node = curr_node.children[child_selected_index]
        
        if curr_node.N <= self.expansion_threshold:
            return curr_node
        else:
            has_expanded = self.expand_node(curr_node, sm)
            if not has_expanded:
                return curr_node
            return np.random.choice(curr_node.children)

    # def rollout(self, leaf: Node):
    #     state = leaf.state
    #     while not self.sm.is_final(state): # Rollout to final state
    #         curr_player, flipped_state, state_was_flipped = self.sm.flip_state(state)
    #         legal_moves = self.sm.get_legal_moves([curr_player, *flipped_state])
    #         action = self.agent.choose_action(flipped_state, legal_moves) 
    #         action = self.sm.flip_action(action, state_was_flipped)
    #         state = self.sm.get_successor(state, action)
    #     winner = self.sm.get_winner(state)
    #     Z = winner
    #     return Z

    def rollout(self, agent, sm: StateManager, leaf: Node):
        state = leaf.state
        while not sm.is_final(state): # Rollout to final state
            curr_player, flipped_state, state_was_flipped = sm.flip_state(state)
            legal_moves = sm.get_legal_moves([curr_player, *flipped_state])
            action = agent.choose_action(flipped_state, legal_moves) 
            action = sm.flip_action(action, state_was_flipped)
            state = sm.get_successor(state, action)
        winner = sm.get_winner(state)
        Z = winner
        return Z

    def rollout_mp(self, args):
        agent, sm, leaf = args
        np.random.seed()
        Z = self.rollout(agent, sm, leaf)
        # q.put(Z)
        return Z


    def backpropagate(self, leaf, Zs):
        if not isinstance(Zs, list):
            Zs = [Zs]
        Zs_len = len(Zs)
        Zs_sum = sum(Zs)
        curr_node = leaf
        curr_node.N += Zs_len

        while curr_node.parent != None: # Propagate upwards until root is reached
            parent = curr_node.parent
            child_index = parent.children.index(curr_node)
            parent.N += Zs_len
            parent.Ns[child_index] += Zs_len
            parent.Es[child_index] += Zs_sum
            curr_node = parent
        
    def get_visit_distribution(self):
        visit_counts = self.root.Ns
        # dist = F.softmax(torch.Tensor(visit_counts), dim=0)
        dist = np.array(visit_counts)/np.sum(visit_counts)
        moves = [child.origin_action for child in self.root.children] # Serves as indices for the final list
        final_dist = np.zeros(len(self.action_space))
        for i, move in enumerate(moves):
            final_dist[move] = dist[i]
        return final_dist


    def visualize(self, depth=-2, state_as_grid=False):
        self.nodes_created = 0
        vis_tree = graphviz.Graph(name='Monte Carlo Tree', filename=f'TREES/monte_carlo_tree_{self.vis_counter}.gv')
        self.vis_counter += 1
        self.visualize_node(vis_tree, self.root, parent_node_name=None, edge_label=None, add_text=self.root.N, depth=depth, state_as_grid=state_as_grid)
        print(f'Nodes visualized: {self.nodes_created}')
        vis_tree.render(view=True)
    
    def visualize_node(self, vis_tree, node, parent_node_name, edge_label, add_text, depth, state_as_grid):
        if depth == -1:
            return
        self.nodes_created += 1
        name= f'node_{np.random.rand()}'
        label = f'{node.state}' if state_as_grid == False else f'{node.state}' #TODO
        if add_text != None:
            label = f'{label}\n{add_text}'
        # if node.state[0] == 1:
        #     vis_tree.attr('node', shape='triangle')
        # elif node.state[0] == 2:
        #     vis_tree.attr('node', shape='invtriangle')
        if node.state[0] == 1:
            vis_tree.node(name=name, label=label, style='filled', color='#000000' if len(node.children) > 0 else '#20bf6b', fillcolor='#ffffff')
        else:
            vis_tree.node(name=name, label=label, style='filled', color='#000000' if len(node.children) > 0 else '#20bf6b', fillcolor='#dddddd')
        if parent_node_name != None:
            vis_tree.edge(tail_name=parent_node_name, head_name=name, label=edge_label)
        
        
        for child, n, e, q, u in zip(node.children, node.Ns, node.Es, node.get_Qs(), node.get_us()):
            value = q + u if node.state[0] == 1 else q - u
            self.visualize_node(vis_tree, child, parent_node_name=name, edge_label=f'{child.origin_action}', add_text=f'{n}\nE: {e} Q: {q: 0.3f}\nV: {value: 0.3f}', depth=depth-1, state_as_grid=state_as_grid)
        

class Roller:
    def __init__(self, sm):
        self.sm = sm
    
    def rollout(self, args):
        # print(args)
        agent, leaf = args
        state = leaf.state
        while not self.sm.is_final(state): # Rollout to final state
            curr_player, flipped_state, state_was_flipped = self.sm.flip_state(state)
            legal_moves = self.sm.get_legal_moves([curr_player, *flipped_state])
            action = agent.choose_action(flipped_state, legal_moves) 
            action = self.sm.flip_action(action, state_was_flipped)
            # print(f'Flipped action: {action}')
            # sm.render_state([curr_player, *flipped_state])
            state = self.sm.get_successor(state, action)
        winner = self.sm.get_winner(state)
        Z = winner
        return Z