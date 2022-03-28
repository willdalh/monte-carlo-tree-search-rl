from statemanagers.state_manager import StateManager
from ai.agent import PreTrainedAgent

import torch
import numpy as np
import matplotlib.pyplot as plt

class TOPP:
    def __init__(self, game_name, model_paths, sm: StateManager, nn_dim, num_duel_games):
        self.game_name = game_name.lower()
        self.agents = [PreTrainedAgent(path, nn_dim) for path in model_paths]
        self.sm = sm
        self.num_games = num_duel_games
        for i, agent in enumerate(self.agents):
            agent.index = i
        self.wins = np.zeros(len(self.agents))

        self.state_counts = {}
        
        print(model_paths)



    def run(self, alternate=True):
        last = -1
        for i, agent1 in enumerate(self.agents[:-1]):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                print(i, j)
                has_rendered = False
        
                for g in range(self.num_games):
                    # render = agent1.index != last and self.game_name == 'hex' and False 
                    # render = agent1.index == 0 and agent2.index == 9
                    render = agent2.index == 9 and not has_rendered
                    switch = False
                    if alternate:
                        # switch = np.random.choice([True, False])
                        switch = g%2 == 0
                    

                    if switch:
                        if render:
                            print(f'{agent1.index} plays first')
                        self.duel(agent1, agent2, render)
                    else:
                        if render:
                            print(f'{agent2.index} plays first')
                        self.duel(agent2, agent1, render)
                    last = agent1.index
                    has_rendered = True
                    

    def duel(self, agent1, agent2, render=False):
        state = self.sm.get_initial_state()

        while not self.sm.is_final(state):
            self.count_state(state) # Debug

            curr_player = state[0]
            player, flipped_state, state_was_flipped = self.sm.flip_state(state)
            
            legal_moves = self.sm.get_legal_moves([player, *flipped_state])
            if curr_player == 1:
                action = agent1.choose_action(flipped_state, legal_moves)

            if curr_player == -1:
                action = agent2.choose_action(flipped_state, legal_moves)

            action = self.sm.flip_action(action, state_was_flipped)

            if render and self.game_name == 'hex':
                self.sm.render_state(state)
            state = self.sm.get_successor(state, action)
        
        if render and self.game_name == 'hex':
            chain = self.sm.get_winner_chain(state)
            self.sm.render_state(state, chain)

        winner = self.sm.get_winner(state)
        if winner == 1:
            self.wins[agent1.index] += 1
        elif winner == -1:
            self.wins[agent2.index] += 1

    def count_state(self, state):
        t_state = tuple(state)
        if t_state not in self.state_counts:
            self.state_counts[t_state] = 0
        
        self.state_counts[t_state] += 1

    def present_results(self):
        # print('State counts')
        # for key, val in self.state_counts.items():
        #     print(f'{list(key)}: {val}')
            
        plt.title('Win frequency')
        plt.bar(np.arange(len(self.agents)), self.wins/(self.num_games * (len(self.agents) - 1)))
        plt.show()

    def play_against(self):
        state = self.sm.get_initial_state()
        while not self.sm.is_final(state):
            print('State:', state)
            curr_player = state[0]
            if curr_player == 1:
                action = self.agents[-1].choose_action(state, self.sm.get_legal_moves(state))
                print('Agent chose:', action + 1 )
            else:
                print('Your turn')
                action = int(input()) - 1
            state = self.sm.get_successor(state, action)
            
        

