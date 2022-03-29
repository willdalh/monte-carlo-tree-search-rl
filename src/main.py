import argparse
import os
import shutil
import json
import importlib
import glob
import logging

from montecarlo.monte_carlo_tree import MonteCarloTree
from ai.agent import Agent
from run_training import run_training

from topp.topp import TOPP

def main(args):
    if not args.run_topp: # Run training

        # Set up logging directory
        if not os.path.isdir('../logs'):
            os.mkdir('../logs')

        not_to_be_deleted = ['../logs/train_log_7x7_good']
        if args.log_dir in not_to_be_deleted:
            raise Exception(f'The log directory {args.log_dir.split("/")[-1]} is not allowed to be deleted')
            
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.mkdir(args.log_dir)
        os.mkdir(f'{args.log_dir}/models')

        with open(f'{args.log_dir}/args.json', 'w') as f: # Save arguments to file
            json.dump(args.__dict__, f, indent=4)

        # Set up logging
        logging.basicConfig(filename=f'{args.log_dir}/debug.log', format='%(message)s', level=logging.DEBUG)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

        # Import state manager as specified by the game argument
        sm_file_name = '%s_state_manager' % args.game.lower()
        sm_class_name = '%sStateManager' % args.game.upper()
        state_manager = importlib.import_module('statemanagers.%s'%sm_file_name).__dict__[sm_class_name]
        sm = state_manager(**vars(args))
        
        # Interpret state and action dimensions and prepare them for use in the agent
        state_size = sm.get_state_size()
        action_space_size = sm.get_action_space_size()
        args.nn_dim.insert(0, state_size-1) # The agent works from the perspective of player 1 and therefore does not need any player id when feeding states to the ANET
        args.nn_dim.append(action_space_size)

        kwargs = vars(args)

        agent = Agent(**kwargs, state_size=state_size-1, action_space_size=action_space_size)
        mct = MonteCarloTree(**kwargs, action_space=sm.get_action_space())

        run_training(args, sm, mct, agent)
    
    else: # Run TOPP
        
        # Load arguments from the saved directory
        saved_args = argparse.Namespace()
        with open(f'{args.saved_dir}/args.json', 'r') as f:
            saved_args.__dict__ = json.load(f)
        
        # Import state manager as specified by the game argument in the saved directory
        sm_file_name = '%s_state_manager' % saved_args.game.lower()
        sm_class_name = '%sStateManager' % saved_args.game.upper()
        state_manager = importlib.import_module('statemanagers.%s'%sm_file_name).__dict__[sm_class_name]
        sm = state_manager(**vars(saved_args))

        # Interpret state and action dimensions and prepare them for use in the agent
        saved_args.nn_dim.insert(0, sm.get_state_size()-1)
        saved_args.nn_dim.append(sm.get_action_space_size())

        # Load all saved models
        model_paths = glob.glob(f'{args.saved_dir}/models/anet*.pt')
        model_paths = sorted(model_paths, key=lambda x: int(x.split('_')[-1][:-3])) # Sort by which episode the model was saved at
        if len(model_paths) < 2:
            raise FileNotFoundError(f'There are not enough saved models in {args.saved_dir} to run the TOPP')
        
        topp = TOPP(saved_dir=args.saved_dir, num_duel_games=args.num_duel_games, alternate=args.alternate, best_starts_first=args.best_starts_first, game=saved_args.game, model_paths=model_paths, sm=sm, nn_dim=saved_args.nn_dim)
        topp.run()
        topp.present_results()


# [i for i in range(N+1) if i%(int(N/(M-1)))==0] N=200, M=5 gives [0, 50, 100, 150, 200]

def str_to_bool(x):
    '''Convert string to boolean'''
    x = x.lower()
    return x == 'true'

def str_to_list(x):
    '''Convert string to list where integers are interpreted as integers and strings as strings'''
    arr = x.split(',')
    return [int(e) if e.isdigit() else e for e in arr]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run main')

    # General parameters
    parser.add_argument('--episodes', type=int, default=500, help='The number of actual games the system will run')

    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='train_log_test', help='The folder name to save logs and instances of ANETs')
    parser.add_argument('--num_anet_saves', type=int, default=10, help='Number of times to save the ANET during the runs')

    # State managers
    parser.add_argument('--game', type=str, default='hex', help='The game to train/play on')
    
    # Hex
    parser.add_argument('--hex_k', type=int, default=5, help='The size of the k x k Hex board')

    # NIM
    parser.add_argument('--nim_n', type=int, default=10, help='The number of pieces the NIM-board starts with')
    parser.add_argument('--nim_k', type=int, default=3, help='The maximum number of pieces a player can remove each round')
    
    # MCTS parameters
    parser.add_argument('--search_games', type=int, default=500, help='The number of search games to be simulated for each root state.')
    parser.add_argument('--search_time', type=float, default=0.5, help='Time allowed for performing search games for each episode. Used when search_games <= 0.')
    parser.add_argument('--c', type=float, default=1.0, help='Exploration constant for the tree policy')

    # ANET and Agent parameters
    parser.add_argument('--buffer_size', type=int, default=500000, help='The maximum size of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of the batch the agent uses to train on')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for the ANET')
    parser.add_argument('--nn_dim', type=str_to_list, default='256,relu,256,relu', help='The structure of the neural network, excluding the state space size at the start and the action space size at the end')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer used by the neural network to perform gradient descent')
    parser.add_argument('--epsilon', type=float, default=1.0, help='The starting epsilon for the epsilon-greedy policy')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='The value to decay epsilon by for every episode')
    parser.add_argument('--pre_trained_path', type=str, default=None, help='Path to a pretrained model to continue training on')

    # TOPP
    parser.add_argument('--run_topp', type=str_to_bool, default=False, help='Whether or not to run TOPP')
    parser.add_argument('--saved_dir', type=str, default='train_log_test', help='The root folder where the saved nets reside')
    parser.add_argument('--num_duel_games', type=int, default=2, help='The number of games to be played between any two ANET-based agents during TOPP')
    parser.add_argument('--alternate', type=str_to_bool, default=False, help='Whether or not to alternate between who starts first during a series of games')
    parser.add_argument('--best_starts_first', type=str_to_bool, default=True, help='If not alternating, whether or not to let the best ANET start first during a series of games')

    # Visualization
    parser.add_argument('--display', type=str_to_bool, default=True, help='Whether or not to display graphs')
    parser.add_argument('--render', type=str_to_bool, default=False, help='Whether or not to render the game')
    parser.add_argument('--frame_delay', type=int, default=0, help='The amount of time to delay between frames in milliseconds. If less than 0, press on spacebar is expected to continue.')


    ns_args = parser.parse_args()
    # Processing of args
    ns_args.log_dir = f'../logs/{ns_args.log_dir}'
    ns_args.saved_dir = f'../logs/{ns_args.saved_dir}'
    ns_args.game = ns_args.game.upper()

    print(f'Arguments:\n{ns_args}\n')
    main(ns_args)



'''

TRY THESE:
python main.py --search_games 0 --search_time 1 --game HEX --hex_k 3 --episodes 100 --num_anet_saves 20 --epsilon_decay 0.99 --lr 0.001 --nn_dim 'conv(c5),relu,400,relu'


FOR DEMO:
python main.py --episodes 100 --hex_k 3 --lr 0.0007 --search_time 1 --search_games 0 --epsilon_decay 0.99 --nn_dim 'conv(c8),relu,conv(c6),relu,100,relu'

python main.py --episodes 40 --hex_k 4 --lr 0.003 --search_time 0.3 --search_games 0 --epsilon_decay 0.99 --nn_dim 'conv(c8),relu,conv(c6),relu,64,relu'
python main.py --episodes 40 --hex_k 4 --lr 0.0025 --search_time 0.3 --search_games 0 --epsilon_decay 0.99 --nn_dim 'conv(c8),relu,conv(c6),relu,64,re lu'
'''
