#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="mcts"
#SBATCH --output=job_output.out
#SBATCH --mail-user=williad@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Starting script"
echo ""
module purge
module load Anaconda3/2020.07
pip install graphviz
pip install pygame
# python main.py --episodes 100 --search_games 500 --hex_k 3 --display False
# python main.py --search_games 0 --search_time 2 --game HEX --hex_k 5 --episodes 600 --num_anet_saves 20 --epsilon_decay 0.992 --lr 0.0009 --nn_dim 412,relu,412,relu --display False
# python main.py --search_games 0 --search_time 2.5 --game HEX --hex_k 7 --episodes 200 --num_anet_saves 20 --epsilon_decay 0.995 --lr 0.0018 --nn_dim 'conv(c3),relu,400,relu'
python main.py --search_games 0 --search_time 5 --game HEX --hex_k 7 --episodes 600 --num_anet_saves 30 --epsilon_decay 0.999 --lr 0.0009 --nn_dim 'conv(c6-k5-p2),relu,conv(c8-k3-p1),relu,400,relu'
uname -a
