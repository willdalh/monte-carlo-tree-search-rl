import argparse
import os
import shutil
import json

def main(args, kwargs):
    print(args)
    
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')

    if os.path.isdir(args.log_folder):
        shutil.rmtree(args.log_folder)
    os.mkdir(ns_args.log_folder)

    with open(f'{ns_args.log_folder}/args.json', 'w') as f:
        json.dump(ns_args.__dict__, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run main')

    parser.add_argument('--log_folder', type=str, default='train_log_test', help='The folder name to save logs and instances of ANETs')

    # Hex
    parser.add_argument('--k', type=int, default=3, help='The size of the k x k Hex board')
    
    ns_args = parser.parse_args()
    # Processing of args
    ns_args.log_folder = f'../logs/{ns_args.log_folder}'

    kwargs = vars(ns_args)

    main(ns_args, kwargs)