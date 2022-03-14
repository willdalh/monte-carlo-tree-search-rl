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
        self.starters = []
        
        print(model_paths)
        
    

    def run(self):
        for i, agent1 in enumerate(self.agents[:-1]):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                self.duel(agent1, agent2, self.num_games, alternate=True)
                print(i, j)

    def duel(self, agent1, agent2, num_games, alternate):
        for g in range(num_games):
            state = self.sm.get_initial_state()
            if alternate:
                who_plays_first = np.random.randint(1, 3)
            else:
                who_plays_first = 1
            self.starters.append(who_plays_first)

            while not self.sm.is_final(state):
                curr_player = state[0]
                legal_moves = self.sm.get_legal_moves(state)

                if curr_player == 1:
                    if who_plays_first == 1:
                        action = agent1.choose_action(state, legal_moves)
                    else:
                        action = agent2.choose_action(state, legal_moves)
                if curr_player == 2:
                    if who_plays_first == 1:
                        action = agent2.choose_action(state, legal_moves)
                    else:
                        action = agent1.choose_action(state, legal_moves)

                state = self.sm.get_successor(state, action)
            winner = self.sm.get_winner(state)

            if who_plays_first == 1:
                if winner == 1:
                    self.wins[agent1.index] += 1
                else:
                    self.wins[agent2.index] += 1
            else:
                if winner == 1:
                    self.wins[agent2.index] += 1
                else:
                    self.wins[agent1.index] += 1

    def present_results(self):
        print('Average who_plays_first:', np.mean(self.starters))
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
            
        

