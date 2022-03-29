from statemanagers.state_manager import StateManager
from ai.agent import PreTrainedAgent

import torch
import numpy as np
import matplotlib.pyplot as plt

class TOPP:
    def __init__(self, saved_dir, num_duel_games, alternate, best_starts_first, game, model_paths, sm: StateManager, nn_dim, **_):
        self.saved_dir = saved_dir
        self.num_games = num_duel_games
        self.alternate = alternate
        self.best_starts_first = best_starts_first
        self.game_name = game.lower()

        self.agents = [PreTrainedAgent(path, nn_dim) for path in model_paths]
        self.episodes_saved_at = [int(e.split('_')[-1][:-3]) for e in model_paths]
        self.sm = sm
        for i, agent in enumerate(self.agents):
            agent.index = i
        self.wins = np.zeros(len(self.agents))

        
        print(model_paths)


    def run(self):
        last = -1
        for i, agent1 in enumerate(self.agents[:-1]):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                # print(i, j)
                who_starts = self.episodes_saved_at[j] if self.best_starts_first else self.episodes_saved_at[i]
                start_policy = "Alternating" if self.alternate else f'{who_starts} starts'
                print(f'Agent {self.episodes_saved_at[i]} vs Agent {self.episodes_saved_at[j]}: {start_policy}')
                has_rendered = False
        
                for g in range(self.num_games):
                    # render = agent1.index != last and self.game_name == 'hex' and False 
                    # render = agent1.index == 0 and agent2.index == 9
                    render = agent2.index == 9 and not has_rendered and False
                    switch = False
                    if self.alternate:
                        # switch = np.random.choice([True, False])
                        switch = g%2 == 0
                    else:
                        switch = self.best_starts_first
                    

                    if render:
                        print('Rendering')
                    if switch:
                        if render: print(f'{j} starts')
                        self.duel(agent2, agent1, render)
                    else:
                        if render: print(f'{i} starts')
                        self.duel(agent1, agent2, render)
                    last = agent1.index
                    has_rendered = True
                    

    def duel(self, agent1, agent2, render=False):
        state = self.sm.get_initial_state()

        while not self.sm.is_final(state):

            curr_player, flipped_state, state_was_flipped = self.sm.flip_state(state)
            
            legal_moves = self.sm.get_legal_moves([curr_player, *flipped_state])
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

    def present_results(self):
        saved_dir_name = self.saved_dir.split('/')[-1]
        plt.title(f'Win frequency for agents in {saved_dir_name} when {"alternating" if self.alternate else ("best starts first" if self.best_starts_first else "worst starts first")}')
        plt.bar([str(e) for e in self.episodes_saved_at], self.wins/(self.num_games * (len(self.agents) - 1)))
        plt.xlabel('Episode saved at')

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
            


