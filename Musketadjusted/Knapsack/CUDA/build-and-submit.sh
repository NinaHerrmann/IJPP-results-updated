#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/Knapsack/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/Knapsack/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/Knapsack/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/Knapsack/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make Knapsack_0 && \
cd ${source_folder} && \

sbatch job.sh
