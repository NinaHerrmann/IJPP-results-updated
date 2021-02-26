#!/bin/bash
# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
nvcc src/Knapsack_0.cu -o build/Knapsack &&\

echo "run; iterations; problem; colony size; Fitness; Total Time "
for KNAP in 6
do
	for RUN in 1
	do
		build/Knapsack $RUN 2 $KNAP
	done
done

