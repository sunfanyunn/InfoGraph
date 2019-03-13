#!/bin/bash -ex

for i in 1 2 3 4 5
do
  for gc in 3 5 8 16
  do

    CUDA_VISIBLE_DEVICES=$1 python deepinfomax.py --DS $2 --lr 0.01 --local --num-gc-layers $gc

    CUDA_VISIBLE_DEVICES=$1 python deepinfomax.py --DS $2 --lr 0.001 --local --num-gc-layers $gc
  done
done

