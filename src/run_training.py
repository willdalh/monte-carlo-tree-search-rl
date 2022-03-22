import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging

from ai.agent import Agent
from statemanagers.state_manager import StateManager
from montecarlo.monte_carlo_tree import MonteCarloTree


def run_training(args, sm: StateManager, mct: MonteCarloTree, agent: Agent):
    NUM_EPISODES = args.episodes
    NUM_SEARCH_GAMES = args.search_games
    NUM_ANET_SAVES = args.num_anet_saves
    tic = time.time()

    for e in range(NUM_EPISODES + 1):
        if e%(int(NUM_EPISODES/(NUM_ANET_SAVES - 1))) == 0:
            agent.anet.save_model(f'{args.log_dir}/models', f'anet_{e}.pt')
        if e == NUM_EPISODES:
            break

        state = sm.get_initial_state()

        mct.set_root(state)
        mct.expand_node(mct.root, sm)

        while not sm.is_final(state):
            # Initialize monte carlo game board to same state as root
            # mct.expand_to_depth(sm)

            if args.search_time <= 0:
                for g in range(NUM_SEARCH_GAMES):
                    # Tree policy
                    leaf = mct.tree_search_expand(sm)
                    Z = mct.rollout(agent, sm, leaf)
                    mct.backpropagate(leaf, Z)
            else:
                start_time = time.time()
                search_games = 0
                while time.time() - start_time < args.search_time:
                    leaf = mct.tree_search_expand(sm)
                    Z = mct.rollout(agent, sm, leaf)
                    mct.backpropagate(leaf, Z)
                    search_games += 1
            
            # mct.visualize()
            # quit()
            D = mct.get_visit_distribution()
            
            agent.store_case((state, D), sm)
            action = np.argmax(D)
            
            child_selected = None
            for child in mct.root.children:
                if action == child.origin_action:
                    child_selected = child
                    break
            mct.set_root(child_selected)
            state = mct.root.state

        
        # Train ANET on a random minibatch of cases from ReplayBuffer
        mean_loss = agent.train_on_buffer_batch(debug=NUM_EPISODES - e < 30)

        
        # print('End of episode')
        if e%(int(NUM_EPISODES/40)) == 0:
            print(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon: 0.5f} loss={mean_loss: 0.4f}')

        # [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]


    toc = time.time()
    print(f'Time used for training: {toc-tic} seconds')

    # Verification
    agent.present_results(args.log_dir)
    agent.epsilon = 0
    for e in range(4):
        logging.debug(f'Round {e}')
        state = sm.get_initial_state()
        while not sm.is_final(state):
            logging.debug(f'Current state: {state}')
            action = agent.choose_action(state, sm.get_legal_moves(state), debug=True)
            if action not in sm.get_legal_moves(state):
                logging.debug('Had to choose random')
                action = np.random.choice(sm.get_legal_moves(state))
            state = sm.get_successor(state, action)
    

