#!/bin/bash
#SBATCH --job-name 70Knapsack
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=1
#SBATCH --partition gpuv100
#SBATCH --exclusive
#SBATCH --error /scratch/tmp/n_herr03/ACO/Knapsack/70knapsack.err
#SBATCH --output /scratch/tmp/n_herr03/ACO/Knapsack/70knapsack.dat
#SBATCH --mail-type ALL
#SBATCH --mail-user n_herr03@uni-muenster.de
#SBATCH --time 7:30:00
#SBATCH --gres=gpu:1

module load gcccuda

echo "run;iterations;problem;ants;OptSolution;totaltime"
for KNAP in 1 3 4 5 6 7
do
    for RUN in 10
    do
      srun /home/n/n_herr03/IJPP-Knapsack/MusketProgrammwithoutfileread/Knapsack/CUDA/build/70knapsack $RUN 50 $KNAP
    done
done

