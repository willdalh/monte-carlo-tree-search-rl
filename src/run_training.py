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
        logging.debug(f'Running episode {e}')
        if e%(int(NUM_EPISODES/(NUM_ANET_SAVES - 1))) == 0:
            agent.anet.save_model(f'{args.log_dir}/models', f'anet_{e}.pt')
        if e == NUM_EPISODES:
            break

        state = sm.get_initial_state()

        study = False
        study_states = [
            [-1, *list(np.array([
                [ 0,  0,  1,  0,  0],
                [ 0,  1, -1,  1,  0],
                [-1, -1, -1, -1,  0],
                [ 1,  1,  0,  0,  0],
                [ 1, -1,  0,  0,  1]
            ]).ravel())],

            [1, *list(np.array([
                [ 0,  0,  1,  0,  0],
                [ 0,  1, -1,  0,  1],
                [-1, -1, -1, -1,  0],
                [ 1,  1,  0,  0,  0],
                [ 1, -1,  0,  0,  0]
            ]).ravel())],

            [1, *list(np.array([
                [-1, -1, 0],
                [0, 0, 0], 
                [0, 1, 1]
            ]).ravel())],

            [-1, *list(np.array([
                [-1, -1, 0],
                [0, 1, 0], 
                [0, 1, 1]
            ]).ravel())]

        ]
        if study:
            state = study_states[3]
            

        mct.set_root(state)
        mct.expand_node(mct.root, sm)

        while not sm.is_final(state):

            if args.search_games > 0:
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
                if e < 5:
                    logging.debug(f'Episode {e} - search games done: {search_games}')

            if study:
                if args.display:
                    mct.visualize(depth=1)
                    # sm.render_state(state)
                print(mct.get_visit_distribution().reshape(sm.K, sm.K))
               
            
            # if args.display:
            #     mct.visualize(depth=1)
            #     sm.render_state(state)
            # count += 1
            # if count == 7:

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
        logging.debug(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon:0.5f} loss={mean_loss:0.4f} buffersize={agent.buffer.cases_added}')
        if e%(int(NUM_EPISODES/40)) == 0:
            print(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon:0.5f} loss={mean_loss:0.4f} buffersize={agent.buffer.cases_added}')

        # [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]


    toc = time.time()
    print(f'Time used for training: {toc-tic} seconds')

    # Verification
    agent.present_results(args.log_dir, args.display)
    agent.epsilon = 0
    for e in range(4):
        logging.debug(f'Round {e}')
        state = sm.get_initial_state()
        while not sm.is_final(state):
            logging.debug(f'Current state: {state}')
            curr_player, flipped_state, state_was_flipped = sm.flip_state(state)
            action = agent.choose_action(flipped_state, sm.get_legal_moves(state), debug=True)
            action = sm.flip_action(action, state_was_flipped)
            if action not in sm.get_legal_moves(state):
                logging.debug('Had to choose random')
                action = np.random.choice(sm.get_legal_moves(state))
            state = sm.get_successor(state, action)
    

