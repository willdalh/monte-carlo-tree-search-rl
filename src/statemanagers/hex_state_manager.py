import numpy as np
import sys
from .state_manager import StateManager

sys.path.append('statemanagers')
from visualizers.hex_visualizer import HEXVisualizer

class HEXStateManager(StateManager):
    def __init__(self, hex_k, **_):
        self.K = hex_k
        self.visualizer = HEXVisualizer(self.K)

        self.temp_board = np.zeros((self.K, self.K))

    def get_initial_state(self):
        # player_turn = np.random.randint(1, 3)
        player_turn = 1
        return [player_turn, *[0 for _ in range(self.K * self.K)]]

    def get_successor_states(self, state, return_moves=False):
        if self.is_final(state):
            return [], [] # No children available
        legal_moves = self.get_legal_moves(state)
        successors = []
        for a in legal_moves:
            successors.append(self.get_successor(state, a))
        if return_moves:
            return successors, legal_moves
        return successors

    def get_successor(self, state, a):
        curr_player = state[0]
        board = state[1:]
        board[a] = curr_player
        next_player = -1 * curr_player
        return [next_player, *board]

    def _fill_temp_board(self, state):
        board = state[1:]
    
        for y in range(self.K):
            for x in range(self.K):
                self.temp_board[y, x] = board[y * self.K + x] 

    def get_winner_chain(self, state):
        self._fill_temp_board(state)

        # Check two borders
        for i in range(self.temp_board.shape[0]):
            if self.temp_board[0, i] == 1: # Player 1 should span all rows
                found_chain, chain = self.flood_check(self.temp_board, (0, i), 1, [])
                if found_chain:
                    return chain

        for i in range(self.temp_board.shape[0]):
            if self.temp_board[i, 0] == -1: # Player 2 should span all columns
                found_chain, chain = self.flood_check(self.temp_board, (i, 0), -1, [])
                if found_chain:
                    return chain

    def is_final(self, state):
        self._fill_temp_board(state)

        # Check two borders
        for i in range(self.temp_board.shape[0]):
            if self.temp_board[0, i] == 1: # Player 1 should span all rows
                found_chain, chain = self.flood_check(self.temp_board, (0, i), 1, [])
                if found_chain:
                    return True

        for i in range(self.temp_board.shape[0]):
            if self.temp_board[i, 0] == -1: # Player 2 should span all columns
                found_chain, chain = self.flood_check(self.temp_board, (i, 0), -1, [])
                if found_chain:
                    return True

        return False

    

    def flood_check(self, board, coord, player, checked):
        r, c = coord
        checked.append(coord)
        
        # Check if other wall is reached
        if player == 1:
            if r == self.K - 1: # Reached last row
                return True, [coord]
        if player == -1:
            if c == self.K - 1: # Reached last column
                return True, [coord]
                
        # Find valid neighbors of coord
        neigbor_indices = []
        for r_off in [-1, 0, 1]:
            for c_off in [-1, 0, 1]:
                if r_off != c_off: # Fits neighborhood scheme
                    new_r, new_c = r + r_off, c + c_off
                    neighbor = (new_r, new_c)
                    if neighbor not in checked and self.is_valid_coord(neighbor) and board[neighbor] == player:
                        neigbor_indices.append(neighbor)
                        
        found_chain = False
        chain = []
        for neighbor in neigbor_indices:
            found_chain, chain = self.flood_check(board, neighbor, player, checked)
            if found_chain:
                chain.append(coord)
                break
        return found_chain, chain


    def is_valid_coord(self, coord):
        r, c = coord
        return r >= 0 and r < self.K and c >= 0 and c < self.K

    
    def get_winner(self, state):
        curr_player = state[0]
        winner = -1 * curr_player
        return winner

    def get_legal_moves(self, state):
        board = state[1:]
        return [i for i in range(len(board)) if board[i] == 0]

    def get_action_space(self):
        return [i for i in range(self.K*self.K)]

    def get_action_space_size(self):
        return self.K*self.K

    def get_state_size(self):
        return self.K*self.K + 1

    def flip_state(self, state):
        '''Flip state to make it from the perspective of player 1'''
        if state[0] == 1:
            return state[0], state[1:], False
        board = np.array(state[1:]).reshape(self.K, self.K)
        board = board.T * -1
        return  state[0], list(board.ravel()), True

    def flip_action(self, action, state_was_flipped):
        if state_was_flipped:
            r, c = action//self.K, action%self.K # Find row and col
            r, c = c, r # Flip
            return r * self.K + c
        return action

    def render_state(self, state, chain=None):
        board = np.array(state[1:]).reshape(self.K, self.K)
        self.visualizer.draw_board(board, player=state[0], chain=chain)
