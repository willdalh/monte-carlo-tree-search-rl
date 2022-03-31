import time
import numpy as np
import torch
import torch.nn.functional as F
from .node import Node
from statemanagers.state_manager import StateManager
import graphviz
import torch.multiprocessing as tmp

class MonteCarloTree:
    def __init__(self, c, action_space, **_):
        '''
        Initialize the Monte Carlo Tree
        
        Args:
            c: Exploration constant
            action_space: List of all possible actions
            **_: Unused keyword arguments
        '''
        self.action_space = action_space
        self.c = c
        self.expansion_threshold = 0

        self.root = None

        self.vis_counter = 0

    
    def set_root(self, state):
        '''
        Set the root of the Monte Carlo Tree. Accepts both a Node and a state as given by a State Manager.

        Args:
            state: list of state values or a Node
        '''
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

    def expand_node(self, parent: Node, sm: StateManager):
        '''
        Expand a node by generating its children and assigning them to the parent.

        Args:
            parent: Node to expand
            sm: StateManager to use for generating successor states
        
        Returns:
            True if the state in the parent is not final, False otherwise
        '''
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


    def tree_search_expand(self, sm: StateManager):
        '''
        Tree traversal as shown by John Levine in https://www.youtube.com/watch?v=UXW2yZndl7U.
        Performs tree search and expands a leaf node if it has been visited less than the expansion threshold.
        
        Args:
            sm: StateManager to use for generating successor states

        Returns:
            A leaf node, either newly expanded or still below the expansion threshold
        '''
        curr_node = self.root
        while len(curr_node.children) != 0:
            # Quality and exploration values as shown in the lecture slides on MCTS https://www.idi.ntnu.no/emner/it3105/lectures/neural/mcts.pdf
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

    def rollout(self, agent, sm: StateManager, leaf: Node):
        '''
        Perform a rollout from a leaf node to a final state.
        
        Args:
            agent: Agent to use for rollouts
            sm: StateManager to use for creating successor states
            leaf: Leaf node containing the state to do rollout from
        
        Returns:
            The evaluation of the final state reached
        '''
        state = leaf.state
        while not sm.is_final(state): 
            curr_player, flipped_state, state_was_flipped = sm.flip_state(state) # Flip to perspective of player 1
            legal_moves = sm.get_legal_moves([curr_player, *flipped_state])
            action = agent.choose_action(flipped_state, legal_moves) 
            action = sm.flip_action(action, state_was_flipped) # Flip action to original perspective
            state = sm.get_successor(state, action)
        winner = sm.get_winner(state)
        Z = winner
        return Z

    def backpropagate(self, leaf, Z):
        '''
        Propagate the evaluation from a leaf node to the root node by updating the total evaluation and number of visits for all nodes on the path.

        Args:
            leaf: Leaf node to propagate evaluation upwards from
            Z: Evaluation of the final state reached from the leaf node
        '''
        curr_node = leaf
        curr_node.N += 1

        while curr_node.parent != None: # Propagate upwards until root is reached
            parent = curr_node.parent
            child_index = parent.children.index(curr_node)
            parent.N += 1
            parent.Ns[child_index] += 1
            parent.Es[child_index] += Z
            curr_node = parent
        
    def get_visit_distribution(self):
        '''
        Get the visit distribution of all children of the root node. Illegal moves gets a value of 0.

        Returns:
            A list of visit fractions from the root for all moves
        '''
        visit_counts = self.root.Ns
        dist = np.array(visit_counts)/np.sum(visit_counts) # Normalize
        final_dist = np.zeros(len(self.action_space)) # List to be filled with visit distributions of the legal actions
        legal_moves = [child.origin_action for child in self.root.children]
        for i, move in enumerate(legal_moves): # Fill the list with visit distributions of the legal actions
            final_dist[move] = dist[i]
        return final_dist


    def visualize(self, depth=-2):
        '''
        Use graphviz to visualize the tree. Used for debugging.

        Args:
            depth: Depth of the tree to visualize. -2 means to visualize the entire tree.
        '''
        self.nodes_created = 0
        vis_tree = graphviz.Graph(name='Monte Carlo Tree', filename=f'TREES/monte_carlo_tree_{self.vis_counter}.gv') 
        self.vis_counter += 1 # Increment counter to avoid overwriting files in the current session
        self.visualize_node(vis_tree, self.root, parent_node_name=None, edge_label=None, add_text=self.root.N, depth=depth)
        print(f'Nodes visualized: {self.nodes_created}')
        vis_tree.render(view=True)
    
    def visualize_node(self, vis_tree, node, parent_node_name, edge_label, add_text, depth):
        '''
        Visualize a node and recursively its children.

        Args:
            vis_tree: Graphviz graph to add the node to
            node: Node to visualize
            parent_node_name: Name of the parent node in the Graphviz graph
            edge_label: Label of the edge going from the parent node to the node
            add_text: Additional text to add to the node label
            depth: Depth of the tree to visualize. -2 means to visualize the entire tree.
        '''
        if depth == -1:
            return

        self.nodes_created += 1
        name= f'node_{np.random.rand() + time.time()}' # Unique name for the node
        label = f'{node.state}'
        if add_text != None:
            label = f'{label}\n{add_text}'
            
        player = node.state[0]
        # Visually distinguish between player 1 and player 2 and add green border if the node is a leaf node
        if player == 1:
            vis_tree.node(name=name, label=label, style='filled', color='#000000' if len(node.children) > 0 else '#20bf6b', fillcolor='#ffffff') 
        else:
            vis_tree.node(name=name, label=label, style='filled', color='#000000' if len(node.children) > 0 else '#20bf6b', fillcolor='#dddddd')
        if parent_node_name != None:
            vis_tree.edge(tail_name=parent_node_name, head_name=name, label=edge_label)
        
        # Recursively visualize the children with some data added to the label
        for child, n, e, q, u in zip(node.children, node.Ns, node.Es, node.get_Qs(), node.get_us()):
            value = q + u if player == 1 else q - u
            self.visualize_node(vis_tree, child, parent_node_name=name, edge_label=f'{child.origin_action}', add_text=f'{n}\nE: {e} Q: {q: 0.3f}\nV: {value: 0.3f}', depth=depth-1)
        