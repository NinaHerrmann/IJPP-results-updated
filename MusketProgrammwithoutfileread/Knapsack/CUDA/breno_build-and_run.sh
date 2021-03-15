#!/bin/bash

# remove files and create folder
#rm -rf -- build && \
#mkdir build && \

# run cmake
#cd build && \
#cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev -D CMAKE_CXX_COMPILER=g++ ../ && \

#make Knapsack_0 && \
echo "run; iterations; problem; colony size; Fitness; Total Time "
for KNAP in 7 6 5 4 3 1
do
	for RUN in 10
	do
		bin/mknap_aco_gpu_ref $RUN 50 $KNAP
	done	
done
