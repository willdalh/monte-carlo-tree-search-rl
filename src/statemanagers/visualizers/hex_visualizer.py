from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np

class HEXVisualizer:
    def __init__(self, hex_k):
        '''
        Initialize a visualizer for the HEX game.

        Args:
            hex_k (int): The size of the board.
        '''
        self.K = hex_k
        
        self.WIDTH = 400
        self.HEIGHT = 600

        self.num = np.sqrt(self.WIDTH ** 2 + self.HEIGHT ** 2)
        self.piece_radius = (self.num/self.K) / 5.8875

        self.WHITE = (255, 255, 255) 
        self.BLACK = (0, 0, 0) 
        
        self.PLAYER1_COLOR = (231, 76, 60)
        self.PLAYER2_COLOR = (47, 54, 64)

        self.GOLD = (225, 177, 44)
        self.PATH_BLUE = (41, 128, 185)
        self.PATH_RED = (192, 57, 43)



        self.diag_indices = self.get_diag_indices()
        self.diag_points = self.diag_indices_to_points(self.diag_indices)

        self.line_pairs = self.get_line_pairs(self.diag_points)
        

        pygame.init()
        pygame.display.set_caption('HEX')
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 16)

        self.screen_initialized = False
    
    def draw_board(self, state, frame_delay=100, chain=None, nodes_text=None):
        '''
        Draw a state of the HEX game.

        Args:
            state: A list representing the board state.
            frame_delay: The number of milliseconds to wait between frames.
            chain: A list of indices representing a winner chain.
            nodes_text: A list of strings to distribute to the nodes to be displayed inside them.
        '''
        player, board = state[0], np.array(state[1:]).reshape(self.K, self.K)
        if not self.screen_initialized:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.screen_initialized = True
        
        # Clear screen
        pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(0, 0, self.WIDTH, self.HEIGHT))

        # Write text
        img1 = self.font.render('Player 1', True, self.PLAYER1_COLOR)
        img2 = self.font.render('Player 2', True, self.PLAYER2_COLOR)
        self.screen.blit(img1, (10, 20))
        self.screen.blit(img2, (10, 40))

        arrow = self.font.render('<-', True, self.BLACK)
        self.screen.blit(arrow, (80, 20 if player == 1 else 40))

        # Draw lines connecting points
        for pair in self.line_pairs:
            from_point, to_point = pair
            pygame.draw.line(self.screen, self.BLACK, from_point, to_point, width=2)

        if chain:
            points = [self.get_point_from_index(index) for index in chain]
            # color = self.PATH_BLUE if board[chain[0]] == 1 else self.PATH_RED # Check winner
            for point, next_point in zip(points[:-1], points[1:]):
                pygame.draw.line(self.screen, self.GOLD, point, next_point, width=10)

        for i, (points, indices) in enumerate(zip(self.diag_points, self.diag_indices)): # Draw points
            for j, (point, index) in enumerate(zip(points, indices)):
                player = board[index]
                index_text = None
                index_flat = index[0] * self.K + index[1]
                text_color = self.BLACK if player == 0 or player == 1 else self.WHITE
                
                # Different style for point based on which player owns it
                if player == 0:
                    pygame.draw.circle(self.screen, self.BLACK, point, self.piece_radius, width=3)
                    pygame.draw.circle(self.screen, self.WHITE, point, self.piece_radius - 3)
                elif player == 1:
                    pygame.draw.circle(self.screen, self.PLAYER1_COLOR, point, self.piece_radius)
                elif player == -1:
                    pygame.draw.circle(self.screen, self.PLAYER2_COLOR, point, self.piece_radius)

                if self.K <= 10:
                    # Add flattened index inside point
                    index_text = self.small_font.render(f'{index_flat}', True, text_color)
                    index_text_rect = index_text.get_rect(center=point if nodes_text == None else (point[0], point[1]-8))
                    self.screen.blit(index_text, index_text_rect)

                    # Add additional text inside nodes if available
                    if nodes_text != None:
                        desc_text = self.small_font.render(f'{nodes_text[index_flat]}', True, text_color)
                        desc_text_rect = desc_text.get_rect(center=(point[0], point[1]+2))
                        self.screen.blit(desc_text, desc_text_rect)
         
        pygame.display.flip()
        if frame_delay < 0: # Press space to continue
            while True:
                break_loop = False
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            break_loop = True
                if break_loop:
                    break
        elif frame_delay > 0:
            pygame.time.wait(frame_delay)
        

    def get_diag_indices(self):
        '''
        Generate a list of indices for the diagonals of the HEX board.

        Returns:
            A list of lists where the ith inner list contains the indices of the points on the ith diagonal.
        '''

        diag_indices = []
        # Iterate over the left and bottom borders
        for b in range(2 * self.K - 1):
            if b < self.K - 1: # Left border
                y, x = b, 0

            elif b == self.K - 1: # Bottom left element
                y, x = b, 0

            elif b >= self.K: # Bottom border
                y, x = self.K - 1, b - self.K + 1
            
            indices = []
            for new_y, new_x in zip(range(y, -1, -1), range(x, self.K)): # ITerate over the diagonal elements starting from the border element
                indices.append((new_y, new_x))
            diag_indices.append(indices)
        
        return diag_indices

    def diag_indices_to_points(self, diag_indices):
        '''
        Generate a list of lists of the screen points corresponding to each index in diag_indices.

        Args:
            diag_indices: A list of lists of indices for the diagonal elements.

        Returns:
            A list of lists where the ith inner list is the screen points corresponding to the indices in the ith inner list.
        '''
        num_diags = len(diag_indices)
        diag_points = []
        for i, indices in enumerate(diag_indices):
            num_pieces = len(indices)
            points = []
            y = i * self.HEIGHT / (num_diags) + (self.HEIGHT / (2 * num_diags)) # Center middle row vertically
            for j, index in enumerate(indices):
                # Distribute points horizontally so the first and last points have the same distance to their vertical borders
                x = j * self.WIDTH / self.K + (self.WIDTH / (2 * self.K)) + (self.K - num_pieces) * (self.WIDTH / (2 * self.K)) 
                points.append((x, y))
            diag_points.append(points)
        return diag_points

    def get_line_pairs(self, diag_points):
        '''
        Generate a list of tuples for each pair of point to create a line between.
        Assumes points are to be connected the way they are in HEX.

        Args:
            diag_points: A list of lists of points.

        Returns:
            A list of point pairs (as tuples).
        '''
        pairs = []
        for points, last_points in zip(diag_points[1:self.K], diag_points[:self.K - 1]): # Lines for top half
            for point, last_point in zip(points[:-1], last_points): # Up-right lines
                pairs.append((point, last_point))
            for point, last_point in zip(points[1:], last_points): # Up-left lines
                pairs.append((point, last_point))
            for point, next_point in zip(points[:-1], points[1:]): # Horizontal lines
                pairs.append((point, next_point))

        for points, last_points in zip(diag_points[self.K:], diag_points[self.K - 1:]): # Lines for bottom half
            for point, last_point in zip(points, last_points[:-1]): # Up-left lines
                pairs.append((point, last_point))
            for point, last_point in zip(points, last_points[1:]): # Up-right lines
                pairs.append((point, last_point))
            for point, next_point in zip(points[:-1], points[1:]): # Horizontal lines
                pairs.append((point, next_point))
        return pairs

    def get_point_from_index(self, index1):
        '''
        Index into the list of screen points to get the point corresponding to the index.

        Args:
            index1: The index of the point.

        Returns:
            The screen point corresponding to the index.
        '''
        for i, indices in enumerate(self.diag_indices):
            for j, index2 in enumerate(indices):
                if index1 == index2:
                    return self.diag_points[i][j]

