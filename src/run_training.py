import numpy as np
import torch
import matplotlib.pyplot as plt
import time

from ai.agent import Agent
from statemanagers.state_manager import StateManager
from montecarlo.monte_carlo_tree import MonteCarloTree


def run_training(args, sm: StateManager, mct: MonteCarloTree, agent: Agent):
    NUM_EPISODES = args.episodes
    NUM_SEARCH_GAMES = args.search_games
    NUM_ANET_SAVES = args.num_anet_saves
    tic = time.time()

    # Store one instance of the ANET prior to training
    agent.anet.save_model(f'{args.log_dir}/models', f'anet_{0}.pt')

    for e in range(NUM_EPISODES + 1):
        state = sm.get_initial_state()

        mct.set_root(state)
        mct.expand_node(mct.root, sm)

        while not sm.is_final(state):
            # Initialize monte carlo game board to same state as root
            # mct.expand_to_depth(sm)

            for g in range(NUM_SEARCH_GAMES):
                # Tree policy
                leaf = mct.tree_search_expand(sm)
                Z = agent.rollout(sm, leaf)
                mct.backpropagate(leaf, Z)
                # mct.visualize()

            D = mct.get_visit_distribution()
            if (state == [1, 2] or state == [2, 2] or state == [1, 3] or state == [2, 3]) and False:
                print('\n')
                print('state:', state)
                print('Qs:', mct.root.get_Qs())
                print('Ns:', mct.root.Ns)
                print('dist:', D)
            # print('\n', state)
            # print('Qs', mct.root.get_Qs())
            # print('us', mct.root.get_us())
            # print('N', mct.root.N)
            
            # mct.visualize()
            # if state[1] != 8:
            #     quit()
            
            agent.store_case((state, D))
            action = np.argmax(D)
            
            child_selected = None
            for child in mct.root.children:
                if action == child.origin_action:
                    child_selected = child
                    break
            mct.set_root(child_selected)
            state = mct.root.state
            # D = distribution of visit counts in MCT along all arcs emanating from root
            # Add case (root, D) Will be (state, D)

            # Choose actual move a* based on D
            # Perform a* on root to produce successor state s*
            # Update mct to s*
            # In MCT, retain subtree rooted at s*, discard everything else
            # set root = s*
        
        # Train ANET on a random minibatch of cases from ReplayBuffer
        agent.train_on_buffer_batch(sm, debug=NUM_EPISODES - e < 30)
        winner = sm.get_winner(state)
        
        # print('End of episode')
        if e%(int(NUM_EPISODES/20)) == 0:
            print(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon}')

        if e%(int(NUM_EPISODES/(NUM_ANET_SAVES - 1))) == 0 and e != 0:
            agent.anet.save_model(f'{args.log_dir}/models', f'anet_{e}.pt')
            # Save ANET parameters

        # [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]






    toc = time.time()
    print(f'Time used for training: {toc-tic} seconds')

    # Verification
    agent.present_results()
    agent.epsilon = 0
    for e in range(4):
        print('Round', e)
        state = sm.get_initial_state()
        while not sm.is_final(state):
            print('Current state:', state)
            action = agent.choose_action(state, sm.get_legal_moves(state), debug=True)
            if action not in sm.get_legal_moves(state):
                print('Had to choose random')
                action = np.random.choice(sm.get_legal_moves(state))
            state = sm.get_successor(state, action)
    

    print('Replay buffer contents:')
    agent.buffer.print_histogram()
    # print('Dist for [1, 3]', output)
    # plt.plot(np.arange(len(wins)), wins)
    # plt.show()

