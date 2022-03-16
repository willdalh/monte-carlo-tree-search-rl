from statemanagers.state_manager import StateManager
from ai.agent import Agent

import torch
import numpy as np
import matplotlib.pyplot as plt

class TOPP:
    def __init__(self, model_paths, sm: StateManager, nn_dim):
        self.agents = [Agent(100, 100, 100, nn_dim, 'adam', 1000) for i in model_paths]
        self.sm = sm
        self.num_games = 50
        for i, (agent, path) in enumerate(zip(self.agents, model_paths)):
            agent.epsilon = 0
            agent.load_model(path)
            agent.index = i
        self.wins = np.zeros(len(self.agents))

        self.state_counts = {}
        
        print(model_paths)
        
    

    def run(self):
        for i, agent1 in enumerate(self.agents[:-1]):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                for g in range(self.num_games):
                    if np.random.randint(1, 3) == 1: 
                        self.duel(agent1, agent2)
                    else:
                        self.duel(agent2, agent1)
                print(i, j)

    def duel(self, agent1, agent2):
        state = self.sm.get_initial_state()
    
        while not self.sm.is_final(state):
            self.count_state(state) # Debug

            curr_player = state[0]
            legal_moves = self.sm.get_legal_moves(state)

            if curr_player == 1:
                action = agent1.choose_action(state, legal_moves)

            if curr_player == 2:
                action = agent2.choose_action(state, legal_moves)

            state = self.sm.get_successor(state, action)

        winner = self.sm.get_winner(state)
        if winner == 1:
            self.wins[agent1.index] += 1
        elif winner == 2:
            self.wins[agent2.index] += 1

    def count_state(self, state):
        t_state = tuple(state)
        if t_state not in self.state_counts:
            self.state_counts[t_state] = 0
        
        self.state_counts[t_state] += 1

    def present_results(self):
        print('State counts')
        for key, val in self.state_counts.items():
            print(f'{list(key)}: {val}')
            
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
            
        

