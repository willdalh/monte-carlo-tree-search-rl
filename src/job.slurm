#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
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
python main.py --log_dir train_log_test --hex_k 3 --episodes 100 --search_games 500 --lr 0.001 --epsilon_decay 0.97 --display False

uname -a
