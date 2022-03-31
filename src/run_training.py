import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging

from ai.agent import Agent
from statemanagers.state_manager import StateManager
from montecarlo.monte_carlo_tree import MonteCarloTree

def run_training(args, sm: StateManager, mct: MonteCarloTree, agent: Agent):
    '''
    Run a training session.

    Args:
        args: The arguments of the training session as a Namespace.
        sm: The state-manager to use.
        mct: The Monte-Carlo-Tree to use for tree-search, node expansion, rollout, and backpropagation.
        agent: The agent to use.
    '''

    NUM_EPISODES = args.episodes
    NUM_SEARCH_GAMES = args.search_games 
    NUM_ANET_SAVES = args.num_anet_saves
    tic = time.time()

    # Outer loop for actual games
    for e in range(NUM_EPISODES + 1):
        logging.debug(f'Running episode {e}')
        if e%(int(NUM_EPISODES/(NUM_ANET_SAVES - 1))) == 0:
            agent.anet.save_model(f'{args.log_dir}/models', f'anet_{e}.pt')
        if e == NUM_EPISODES:
            break

        state = sm.get_initial_state()
        mct.set_root(state)
        mct.expand_node(mct.root, sm)

        # Second loop for actual moves
        while not sm.is_final(state):
            if args.render:
                if e in args.episodes_to_render or args.episodes_to_render[0] == -1:
                    sm.render_state(state, frame_delay=args.frame_delay)
            
            # Inner loop for search games
            if args.search_games > 0:
                for g in range(NUM_SEARCH_GAMES):
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
            
            D = mct.get_visit_distribution()
            
            curr_player, flipped_state, state_was_flipped = sm.flip_state(state) # Flip state to be from the perspective of player 1
            flipped_D = sm.flip_distribution(D, state_was_flipped) # If state was flipped, flip the distribution

            # Create new cases from the current case and store them
            total_cases = [(flipped_state, flipped_D)] + sm.get_symmetric_flipped_cases((flipped_state, flipped_D)) 
            for case in total_cases:
                agent.store_case(case)

            action = np.argmax(D)
            
            # Find the node corresponding to the action taken from the root
            child_selected = None
            for child in mct.root.children:
                if action == child.origin_action:
                    child_selected = child
                    break
            
            # Retain the subtree rooted at the selected child
            mct.set_root(child_selected)
            state = mct.root.state
        
        if args.render:
            if e in args.episodes_to_render or args.episodes_to_render[0] == -1:
                if args.game == 'HEX':
                    chain = sm.get_winner_chain(state)
                    sm.render_state(state, frame_delay=args.frame_delay, chain=chain)
                if args.game == 'NIM':
                    sm.render_state(state)

        # Train ANET on a random minibatch of cases from ReplayBuffer
        mean_loss = agent.train_on_buffer_batch(debug=NUM_EPISODES - e < 30)
        agent.decay_epsilon()

        logging.debug(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon:0.5f} loss={mean_loss:0.4f} buffersize={agent.buffer.cases_added}')
        if e%(int(NUM_EPISODES/40)) == 0:
            print(f'Episode {e}/{NUM_EPISODES}: epsilon={agent.epsilon:0.5f} loss={mean_loss:0.4f} buffersize={agent.buffer.cases_added}')



    toc = time.time()
    print(f'Time used for training: {toc-tic} seconds')

    # Verification
    agent.present_results(args.log_dir, args.display)

    

