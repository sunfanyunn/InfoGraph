#!/bin/bash -ex
for target in 0 1 2 3 4 5 6 7 8 9 10 11
do
  CUDA_VISIBLE_DEVICES=$1 python3 main.py --target $target $2 $3 $4 $5
done
