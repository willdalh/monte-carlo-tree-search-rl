from statemanagers.state_manager import StateManager
from ai.agent import PreTrainedAgent

import torch
import numpy as np
import matplotlib.pyplot as plt

class TOPP:
    def __init__(self, saved_dir, num_duel_games, alternate, best_starts_first, render, frame_delay, game, model_paths, sm: StateManager, nn_dim, **_):
        '''
        Initialize an object for running a Tournament of Progressive Policies.

        Args:
            saved_dir: The directory where the agents are saved.
            num_duel_games: The number of games to play in each duel.
            alternate: Whether to alternate starting agents for each duel.
            best_starts_first: Whether the best in a duel should start.
            render: Whether to render the game.
            frame_delay: The delay between frames in milliseconds. -1 for spacebar press to continue.
            game: The name of the game.
            model_paths: The paths to the models to load.
            sm: The state manager for the game.
            nn_dim: The neural network structure for the saved agents.
        '''
        self.saved_dir = saved_dir
        self.num_games = num_duel_games
        self.alternate = alternate
        self.best_starts_first = best_starts_first
        self.render = render
        self.frame_delay = frame_delay
        self.game_name = game.lower()

        self.agents = [PreTrainedAgent(path, nn_dim) for path in model_paths] # Load agents
        self.episodes_saved_at = [int(e.split('_')[-1][:-3]) for e in model_paths] # Store the episode numbers the agents were saved at
        self.sm = sm

        self.wins = np.zeros(len(self.agents))
        for i, agent in enumerate(self.agents):
            agent.index = i # Index to be used for the win-list

        print(f'Loaded {len(self.agents)} agents.')
        # self.play_against()


    def run(self):
        '''
        Run the Tournament of Progressive Policies.
        '''
        for i, agent1 in enumerate(self.agents[:-1]):
            for j, agent2 in enumerate(self.agents[i+1:], i+1):
                who_starts_when_no_alternation = self.episodes_saved_at[j] if self.best_starts_first else self.episodes_saved_at[i]
                start_policy = "Alternating" if self.alternate else f'{who_starts_when_no_alternation} starts'
                print(f'Agent {self.episodes_saved_at[i]} vs Agent {self.episodes_saved_at[j]}: {start_policy}')
        
                for g in range(self.num_games):
                    render = self.render and agent2.index == self.agents[-1].index

                    switch = False
                    if self.alternate:
                        switch = g%2 == 0
                    else:
                        switch = self.best_starts_first
                    

                    if render:
                        print('Rendering')
                    if switch:
                        self.duel(agent2, agent1, render)
                    else:
                        self.duel(agent1, agent2, render)
                    

    def duel(self, agent1, agent2, render=False):
        '''
        Run a single duel between two agents and update the win-list.
        
        Args:
            agent1: The first agent. This one starts.
            agent2: The second agent.
            render: Whether to render the game if the game is HEX.
        '''
        state = self.sm.get_initial_state()

        while not self.sm.is_final(state):

            curr_player, flipped_state, state_was_flipped = self.sm.flip_state(state) # Flip state to the perspective of player 1
            
            legal_moves = self.sm.get_legal_moves([curr_player, *flipped_state])

            if curr_player == 1:
                action = agent1.choose_action(flipped_state, legal_moves)
            if curr_player == -1:
                action = agent2.choose_action(flipped_state, legal_moves)

            action = self.sm.flip_action(action, state_was_flipped) # Flip action back to original perspective

            if render and self.game_name == 'hex':
                self.sm.render_state(state, frame_delay=self.frame_delay)
            state = self.sm.get_successor(state, action)
        
        if render and self.game_name == 'hex':
            chain = self.sm.get_winner_chain(state)
            self.sm.render_state(state, chain=chain, frame_delay=self.frame_delay)

        # Document winner
        winner = self.sm.get_winner(state)
        if winner == 1:
            self.wins[agent1.index] += 1
        elif winner == -1:
            self.wins[agent2.index] += 1

    def present_results(self):
        '''
        Present the result of the TOPP.
        '''
        saved_dir_name = self.saved_dir.split('/')[-1]
        plt.title(f'Win frequency for agents in {saved_dir_name} when {"alternating" if self.alternate else ("best starts first" if self.best_starts_first else "worst starts first")}')
        plt.bar([str(e) for e in self.episodes_saved_at], self.wins/(self.num_games * (len(self.agents) - 1)))
        plt.xlabel('Episode saved at')
        plt.show()

    def play_against(self):
        '''Play against the last loaded agent. For fun and debugging.'''
        state = self.sm.get_initial_state()
        self.sm.render_state(state, frame_delay=-1)
        while not self.sm.is_final(state):
            print('State:', state)
            curr_player = state[0]
            if curr_player == 1:
                player, flipped_state, state_was_flipped = self.sm.flip_state(state)
                action = self.agents[-1].choose_action(flipped_state, self.sm.get_legal_moves(flipped_state))
                action = self.sm.flip_action(action, state_was_flipped)
                print('Agent chose:', action)
            else:
                print('Your turn')
                action = int(input('Type your action'))
            state = self.sm.get_successor(state, action)
            self.sm.render_state(state, frame_delay=-1)
            


