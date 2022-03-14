import argparse
import os
import shutil
import json
import importlib
import glob

from montecarlo.monte_carlo_tree import MonteCarloTree
from ai.agent import Agent
from run_training import run_training

from topp.topp import TOPP


# Temporary
from ai.model import Model
import torch
import torch.nn.functional as F

def main(args):
    if not args.run_topp:
        # Set up logging directory
        if not os.path.isdir('../logs'):
            os.mkdir('../logs')

        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.mkdir(ns_args.log_dir)
        os.mkdir(f'{ns_args.log_dir}\\models')

        with open(f'{ns_args.log_dir}/args.json', 'w') as f:
            json.dump(ns_args.__dict__, f, indent=2)

        # Import chosen state manager
        sm_file_name = '%s_state_manager' % args.game.lower()
        sm_class_name = '%sStateManager' % args.game
        state_manager = importlib.import_module('statemanagers.%s'%sm_file_name).__dict__[sm_class_name]
        sm = state_manager(**vars(args))

        args.nn_dim.insert(0, sm.get_state_space_size())
        args.nn_dim.append(sm.get_action_space_size())

        kwargs = vars(args)

        agent = Agent(**kwargs)
        mct = MonteCarloTree(**kwargs, action_space=sm.get_action_space())

        run_training(args, sm, mct, agent)
    
    else:
        saved_args = argparse.Namespace()
        with open(f'{ns_args.saved_dir}/args.json', 'r') as f:
            saved_args.__dict__ = json.load(f)
        
        sm_file_name = '%s_state_manager' % saved_args.game.lower()
        sm_class_name = '%sStateManager' % saved_args.game
        state_manager = importlib.import_module('statemanagers.%s'%sm_file_name).__dict__[sm_class_name]
        sm = state_manager(**vars(saved_args))

        saved_args.nn_dim.insert(0, sm.get_state_space_size())
        saved_args.nn_dim.append(sm.get_action_space_size())
  
        model_paths = glob.glob(f'{args.saved_dir}/models/anet*.pt')
        model_paths = sorted(model_paths, key=lambda x: int(x.split('_')[-1][:-3]))
        if len(model_paths) == 0:
            raise FileNotFoundError('No saved ANETs found in the specified directory')

        topp = TOPP(model_paths, sm, nn_dim=saved_args.nn_dim)
        topp.run()
        topp.present_results()


# [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]

def str_to_bool(x):
    x = x.lower()
    return x == 'true'

def str_to_list(x):
    arr = x.split(',')
    return [int(e) if e.isdigit() else e for e in arr]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run main')

    # General parameters
    parser.add_argument('--episodes', type=int, default=50, help='The number of actual games the system will run')

    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='train_log_test', help='The folder name to save logs and instances of ANETs')
    parser.add_argument('--num_anet_saves', type=int, default=10, help='Number of times to save the ANET during the runs')

    # State managers
    parser.add_argument('--game', type=str, default='NIM', help='The game to train/play on')
    
    # Hex
    parser.add_argument('--hex_k', type=int, default=3, help='The size of the k x k Hex board')

    # NIM
    parser.add_argument('--nim_n', type=int, default=8, help='The number of pieces the NIM-board starts with')
    parser.add_argument('--nim_k', type=int, default=2, help='The maximum number of pieces a player can remove each round')
    
    # MCTS parameters
    parser.add_argument('--search_games', type=int, default=400, help='The number of search games to be simulated for each root state')
    parser.add_argument('--max_depth', type=int, default=3, help='The depth that the Monte Carlo Tree should be maintained at')
    parser.add_argument('--c', type=float, default=0.9, help='Exploration constant for the tree policy')

    # ANET and Agent parameters
    parser.add_argument('--buffer_size', type=int, default=5000, help='The maximum size of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of the batch the agent uses to train on')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for the ANET')
    parser.add_argument('--nn_dim', type=str_to_list, default='64,relu,32,relu', help='The structure of the neural network, excluding the state space size at the start and the action space size at the end')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer used by the neural network to perform gradient descent')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='The value to decay epsilon by for every action taken')

    # TOPP
    parser.add_argument('--run_topp', type=str_to_bool, default=False, help='Whether or not to run TOPP')
    parser.add_argument('--saved_dir', type=str, default='train_log_test', help='The root folder where the saved nets reside')
    parser.add_argument('--num_duel_games', type=int, default=10, help='The number of games to be played between any two ANET-based agents during TOPP')




    ns_args = parser.parse_args()
    # Processing of args
    ns_args.log_dir = f'../logs/{ns_args.log_dir}'
    ns_args.saved_dir = f'../logs/{ns_args.saved_dir}'

    print(f'Arguments:\n{ns_args}\n')

    main(ns_args)

    '''
    NIM:

    python main.py --episodes 400 --nn_dim 32,relu,16,relu --epsilon_decay 0.999 --lr 0.001 --batch_size 64
    
    python main.py --episodes 500 --batch_size 64 --lr 0.013 --epsilon_decay 0.9999
    python main.py --episodes 500 --batch_size 64 --lr 0.0013 --epsilon_decay 0.9999
    python main.py --episodes 350 --batch_size 64 --search_games 1000 --lr 0.002 --epsilon_decay 0.9999
    '''