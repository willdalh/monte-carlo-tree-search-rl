import numpy as np
from .state_manager import StateManager

class HEXStateManager(StateManager):
    def __init__(self, hex_k, **_):
        self.K = hex_k

    def get_initial_state(self):
        player_turn = np.random.randint(1, 3)
        return [player_turn, *[0 for _ in range(self.K * self.K)]]

    def get_successor_states(self, state, return_moves=False):
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
        next_player = 2 if curr_player == 1 else 1
        return [next_player, *board]

    def is_final(self, state):
        board = state[1:]
        board = np.array(board).reshape(self.K, self.K)

        # Check two borders
        for i in range(board.shape[0]):
            if board[i, 0] == 1:
                found_chain = self.flood_check(board, (i, 0), 1, [])
                if found_chain:
                    return True

        for i in range(board.shape[0]):
            if board[0, i] == 2:
                found_chain = self.flood_check(board, (0, i), 2, [])
                if found_chain:
                    return True
        
        return False

    def flood_check(self, board, coord, player, checked):
        r, c = coord
        checked.append(coord)
        
        # Check if other wall is reached
        if player == 1:
            if c == self.K - 1:
                return True
        if player == 2:
            if r == self.K - 1:
                return True

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
        for neighbor in neigbor_indices:
            if self.flood_check(board, neighbor, player, checked):
                found_chain = True
                break
        return found_chain


    def is_valid_coord(self, coord):
        r, c = coord
        return r >= 0 and r < self.K and c >= 0 and c < self.K

    
    def get_winner(self, state):
        curr_player = state[0]
        winner = 2 if curr_player == 1 else 1
        return winner

    def get_legal_moves(self, state):
        board = state[1:]
        return [i for i in range(len(board)) if board[i] == 0]

    def get_action_space(self):
        return [i for i in range(self.K*self.K)]

    def get_action_space_size(self):
        return self.K*self.K

    def get_state_space_size(self):
        return self.K*self.K + 1
