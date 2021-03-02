#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev -D CMAKE_CXX_COMPILER=g++ ../ && \

make Knapsack_0 && \
echo "run; iterations; problem; colony size; Fitness; Total Time "
for KNAP in 1 3 4 5 6 7 
do
	for RUN in 1
	do
		bin/Knapsack_0 $RUN 10 $KNAP
	done	
done