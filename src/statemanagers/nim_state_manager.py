import numpy as np
from .state_manager import StateManager

class NIMStateManager(StateManager):
    def __init__(self, nim_n, nim_k, **_):
        '''
        Initialize a NIM state manager.
        
        Args:
            nim_n: The number of stones.
            nim_k: The max number of stones that can be removed at a step.
            **_: Ignored.
        '''

        self.N = nim_n
        self.K = nim_k

    def get_initial_state(self):
        '''Return the initial state.'''
        player_turn = 1
        return [player_turn, self.N]

    def get_successor_states(self, state, return_moves=False):
        '''
        Get the successor states of the given state.
        
        Args:
            state: A list of length 2
            return_moves: If True, return the action indices associated with the successor states.

        Returns:
            successors: A list of successor states.
            legal_moves: A list of legal moves if return_moves is True, None otherwise.
        '''
        legal_moves = self.get_legal_moves(state)
        successors = []
        for a in legal_moves:
            successors.append(self.get_successor(state, a))
        
        if return_moves:
            return successors, legal_moves
        return successors
            
    def get_successor(self, state, a):
        '''
        Get the successor state of the given state with the given action.
        
        Args:
            state: A list of length 2
            a: An action index.
        
        Returns:
            The successor state.
        '''
        curr_player = state[0]
        curr_n = state[1]
        curr_n -= a + 1 # A is the action index
        next_player = -1 * curr_player
        return [next_player, curr_n]

    def is_final(self, state):
        '''Check if the given state is a final state.'''
        return state[1] == 0
    
    def get_winner(self, state):
        '''Get the winner of the given state, assuming it is a final state.'''
        curr_player = state[0]
        winner = -1 * curr_player # Winner is not the current player
        return winner

    def get_legal_moves(self, state):
        '''Get the legal moves from the given state.'''
        curr_n = state[1]
        return list(range(min(self.K, curr_n)))

    def get_action_space(self):
        '''Get the action space of the game.'''
        return list(range(1, min(self.K, self.N) + 1))

    def get_action_space_size(self):
        '''Get the size of the action space.'''
        return len(self.get_action_space())

    def get_state_size(self):
        '''Get the size of states.'''
        return 2

    def flip_state(self, state):
        '''
        Flip state to make it from the perspective of player 1.
        This is achieved by transposing the board and negating it.

        Args:
            state: A list of length K*K + 1

        Returns:
            A tuple of the player id, the flipped state, and a boolean indicating if the state was flipped.
        '''
        return state[0], [state[1]], state[0] == -1

    def flip_action(self, action, state_was_flipped):
        '''Return the action as is'''
        return action

    def flip_distribution(self, D, state_was_flipped):
        '''Return the distribution as is'''
        return D

    def render_state(self, state):
        '''Render the given state.'''
        print(f'State is {state}')
    

