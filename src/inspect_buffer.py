import torch
import numpy
from statemanagers.visualizers.hex_visualizer import HEXVisualizer
import pygame

if __name__ == '__main__':
    vis = HEXVisualizer(7)
    path = '../logs/train_log_7x7_idun'

    states = torch.load(f'{path}/buffer_states.pt')
    targets = torch.load(f'{path}/buffer_targets.pt')
    # indices = range(7000, 8000, 1)
    indices = range(0, 1000, 2)

    for index in indices:
    # index = 7004
        state = [1, *states[index].tolist()]
        target = targets[index].tolist()

        vis.draw_board(state, nodes_text=[f'{e:0.2}'[1:] for e in target])

        while True:
            break_loop = False
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        break_loop = True
            if break_loop:
                break


