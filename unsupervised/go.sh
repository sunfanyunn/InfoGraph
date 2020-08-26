#!/bin/bash -ex

for i in 1 2 3 4 5
do
  for gc in 3 5 8 16
  do

    python main.py --DS $1 --lr 0.01 --local --num-gc-layers $gc

    python main.py --DS $1 --lr 0.001 --local --num-gc-layers $gc
  done
done

