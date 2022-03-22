import numpy as np
from .state_manager import StateManager

class NIMStateManager(StateManager):
    def __init__(self, nim_n, nim_k, **_):
        self.N = nim_n
        self.K = nim_k

    def get_initial_state(self):
        '''Return an initial state'''
        # player_turn = np.random.randint(1, 3)
        player_turn = 1
        return [player_turn, self.N]

    def get_successor_states(self, state, return_moves=False):
        ''''''
        legal_moves = self.get_legal_moves(state)
        successors = []
        for a in legal_moves:
            successors.append(self.get_successor(state, a))
        
        if return_moves:
            return successors, legal_moves
        return successors
            
    def get_successor(self, state, a):
        curr_player = state[0]
        curr_n = state[1]
        curr_n -= a + 1 # Because a is the action index, and choosing index 0 you pick 1 stone
        next_player = -1 * curr_player
        return [next_player, curr_n]

    def is_final(self, state):
        return state[1] == 0
    
    def get_winner(self, state): # Assumes state input is final
        curr_player = state[0]
        winner = -1 * curr_player # Winner is not the current player
        return winner

    def get_legal_moves(self, state):
        curr_n = state[1]
        return list(range(min(self.K, curr_n)))

    def get_action_space(self):
        return list(range(1, min(self.K, self.N) + 1))

    def get_action_space_size(self):
        return len(self.get_action_space())

    def get_state_size(self):
        return 2

    def flip_state(self, state):
        return state[0], [state[1]], True

    def flip_action(self, action, state_was_flipped):
        return action

    # def get_symmetric_cases(self, case):
    #     state, 

    def render_state(self, state):
        print(f'State is {state}')
    

