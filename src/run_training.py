import numpy as np
import torch
import matplotlib.pyplot as plt

from ai.agent import Agent
from statemanagers.state_manager import StateManager
from montecarlo.monte_carlo_tree import MonteCarloTree


def run_training(args, sm: StateManager, mct: MonteCarloTree, agent: Agent):
    NUM_EPISODES = args.episodes
    NUM_SEARCH_GAMES = args.search_games
    NUM_ANET_SAVES = args.num_anet_saves

    # mcgb = copy.deepcopy(mct)
    wins = []
    for e in range(NUM_EPISODES):
        state = sm.get_initial_state()
        if e == 0:
            print('Initial dist on start state:', agent.anet(torch.Tensor([state])))
        mct.set_root(state)

        while not sm.is_final(state):
            # Initialize monte carlo game board to same state as root
            # print('Expanding nodes')
            # print('State is', mct.root.state)
            # print('Is final', sm.is_final())

            mct.expand_to_depth(sm, depth=4)
            # mcgb.set_root(state)

            for g in range(NUM_SEARCH_GAMES):
                # Tree policy
                # Use tree policy to search from root to a leaf. Update mcgb with each move
                leaf = mct.tree_search()
                # print('Tree search performed')
                Z = agent.rollout(sm, leaf)
                # print('Rollout performed')
                mct.backpropagate(leaf, Z)
                # print('Backpropagate performed')

            D = mct.get_visit_distribution()
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
        agent.train_on_buffer_batch()
        winner = sm.get_winner(state)
        wins.append(1 if winner == 1 else 0)

        if e%(int(NUM_EPISODES/20)) == 0:
            print(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon}')

        if e%(int(NUM_EPISODES/(NUM_ANET_SAVES - 1))) == 0:
            agent.anet.save_model(f'{args.log_dir}/models', f'anet{e}.pt')
            # Save ANET parameters

        # [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]
    
    # Verification
    for e in range(4):
        print('Round', e)
        state = sm.get_initial_state()
        while not sm.is_final(state):
            print('Current state:', state)
            output = agent.anet(torch.Tensor([state]))
            print('dist:', output)
            action = torch.argmax(output).item()
            state = sm.get_successor(state, action)

    
    # print('Dist for [1, 3]', output)
    plt.plot(np.arange(len(wins)), wins)
    plt.show()

