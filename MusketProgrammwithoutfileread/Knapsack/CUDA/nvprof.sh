#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/Knapsack/CUDA/out/ && \
rm -rf -- ~/build/mnp/Knapsack/cuda && \
mkdir -p ~/build/mnp/Knapsack/cuda && \

# run cmake
cd ~/build/mnp/Knapsack/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make Knapsack_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
