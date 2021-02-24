#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/Knapsack/GPU/out/ && \
rm -rf -- ~/build/mnp/Knapsack/gpu && \
mkdir -p ~/build/mnp/Knapsack/gpu && \

# run cmake
cd ~/build/mnp/Knapsack/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make Knapsack_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
