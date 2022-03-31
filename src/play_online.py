import argparse
import json
from dotenv import load_dotenv
import os

from online.OnlineBot import OnlineBot

def main(args, saved_args):
    load_dotenv()
    token = os.getenv('ONLINE_TOKEN')

    bot = OnlineBot(model_path=f'../logs/{args.saved_dir}/models/{args.model_name}', nn_dim=saved_args.nn_dim, token=token)
    for i in range(1000):
        # load_dotenv()
        # should_stop = os.getenv('SHOULD_STOP').lower() == 'true'
        # if should_stop: 
        #     print('Safely disconnected')
        #     break

        bot.run() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run online play')

    # General parameters
    parser.add_argument('--saved_dir', type=str, default='None', help='The directory of the training session where the saved net is')
    parser.add_argument('--model_name', type=str, default='None', help='The name of the model')

    args = parser.parse_args()

    saved_args = argparse.Namespace()
    with open(f'../logs/{args.saved_dir}/args.json', 'r') as f:
        saved_args.__dict__ = json.load(f)
    
    saved_args.nn_dim.insert(0, 49) 
    saved_args.nn_dim.append(49)

    main(args, saved_args)

    
